#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"
#include "GPUReconstruct.cuh"
#include "GPUStringUnary.cuh"
#include "MaybeDeref.cuh"

/// Kernel for prediction of strings lengths after cutting (for LEFT or RIGHT operations)
template <typename T>
__global__ void
kernel_predict_length_cut(int32_t* newLengths, GPUMemory::GPUString inCol, int32_t stringCount, T lengthLimit, int32_t numberCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    const bool stringCol = (stringCount > 1);
    for (int32_t i = idx; i < (stringCol ? stringCount : numberCount); i += stride)
    {
        const int32_t length = GetStringLength(inCol.stringIndices, stringCol ? i : 0);
        newLengths[i] = min(length, static_cast<int32_t>(maybe_deref(lengthLimit, i)));
    }
}

/// Kernel for string cutting (LEFT or RIGHT operations) - final char copying
template <typename OP>
__global__ void kernel_string_cut(GPUMemory::GPUString outCol,
                                  int32_t* newLengths,
                                  int32_t outStringCount,
                                  GPUMemory::GPUString inCol,
                                  int32_t inStringCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    const int32_t cycleCount = max(outStringCount, inStringCount);
    for (int32_t i = idx; i < cycleCount; i += stride)
    {
        const int32_t inI = (inStringCount > 1) ? i : 0;
        const int32_t outI = (outStringCount > 1) ? i : 0;
        for (int32_t j = 0; j < newLengths[outI]; j++)
        {
            outCol.allChars[GetStringIndex(outCol.stringIndices, outI) + j] =
                inCol.allChars[GetStringIndex(inCol.stringIndices, inI) + j +
                               OP{}(newLengths[outI], inCol.stringIndices, inI)];
        }
    }
}

__global__ void kernel_predict_length_concat(int32_t* newLengths,
                                             GPUMemory::GPUString inputA,
                                             bool isACol,
                                             GPUMemory::GPUString inputB,
                                             bool isBCol,
                                             int32_t dataElementCount);

__global__ void kernel_string_concat(GPUMemory::GPUString output,
                                     GPUMemory::GPUString inputA,
                                     bool isACol,
                                     GPUMemory::GPUString inputB,
                                     bool isBCol,
                                     int32_t dataElementCount);

/// Namespace for string binary operations
namespace StringBinaryOperations
{
struct left
{
    typedef std::string RetType;
    __device__ int32_t operator()(int32_t newLength, int64_t* oldIndices, const int32_t i) const
    {
        return 0;
    }
};

struct right
{
    typedef std::string RetType;
    __device__ int32_t operator()(int32_t newLength, int64_t* oldIndices, const int32_t i) const
    {
        return GetStringLength(oldIndices, i) - newLength;
    }
};

struct concat
{
    typedef std::string RetType;
};
} // namespace StringBinaryOperations


/// Class for all string binary operations
class GPUStringBinary
{
public:
    /// Run generic binary string operation
    /// Pre-computes lengths of output strings and do the operation
    /// <param name="outCol">output string column</param>
    /// <param name="ACol">input string column</param>
    /// <param name="stringCount">count of strings in ACol</param>
    /// <param name="BCol">input column with numbers</param>
    /// <param name="numberCount">count of numbers in BCol</param>
    template <typename OP, typename T>
    static void
    Run(GPUMemory::GPUString& outCol, GPUMemory::GPUString ACol, int32_t stringCount, T BCol, int32_t numberCount)
    {
        // Check counts (have to be 1:1, 1:n, n:1 or n:n
        if (stringCount < 1 || numberCount < 1)
        {
            throw std::runtime_error("Zero or negative data element count");
        }
        if (stringCount > 1 && numberCount > 1 && stringCount != numberCount)
        {
            throw std::runtime_error(
                "String and number count must be the same for col-col operations");
        }

        Context& context = Context::getInstance();
        int32_t outputCount = max(stringCount, numberCount);

        // Predict new lengths
        cuda_ptr<int32_t> newLengths(outputCount);
        kernel_predict_length_cut<<<context.calcGridDim(outputCount), context.getBlockDim()>>>(
            newLengths.get(), ACol, stringCount, BCol, numberCount);

        // Alloc and compute new stringIndices
        GPUMemory::alloc(&(outCol.stringIndices), outputCount);
        GPUReconstruct::PrefixSum(outCol.stringIndices, newLengths.get(), outputCount);

        // Get total char count and alloc allChars
        int64_t outTotalCharCount;
        GPUMemory::copyDeviceToHost(&outTotalCharCount, outCol.stringIndices + outputCount - 1, 1);
        GPUMemory::alloc(&(outCol.allChars), outTotalCharCount);

        // Copy appropriate part of strings
        kernel_string_cut<OP>
            <<<context.calcGridDim(outputCount), context.getBlockDim()>>>(outCol, newLengths.get(),
                                                                          outputCount, ACol, stringCount);

        CheckCudaError(cudaGetLastError());
    }

    template <typename OP>
    static void Run(GPUMemory::GPUString& output,
                    GPUMemory::GPUString inputA,
                    bool isACol,
                    GPUMemory::GPUString inputB,
                    bool isBCol,
                    int32_t dataElementCount)
    {
        static_assert(std::is_same<OP, StringBinaryOperations::concat>::value,
                      "Operation not implemented for String-String.");

        // Predict new lengths
        Context& context = Context::getInstance();
        cuda_ptr<int32_t> newLengths(dataElementCount);
        kernel_predict_length_concat<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            newLengths.get(), inputA, isACol, inputB, isBCol, dataElementCount);

        // Alloc and compute new stringIndices
        GPUMemory::alloc(&(output.stringIndices), dataElementCount);
        GPUReconstruct::PrefixSum(output.stringIndices, newLengths.get(), dataElementCount);

        // Get total char count and alloc allChars
        int64_t outTotalCharCount;
        GPUMemory::copyDeviceToHost(&outTotalCharCount, output.stringIndices + dataElementCount - 1, 1);
        GPUMemory::alloc(&(output.allChars), outTotalCharCount);

        // Concat the strings
        kernel_string_concat<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            output, inputA, isACol, inputB, isBCol, dataElementCount);
    }

    /// Binary string operation string column - string column, output is also string column
    /// <param name="output">output string column</param>
    /// <param name="ACol">first input string column (GPUString)</param>
    /// <param name="BCol">second input string column (GPUString)</param>
    /// <param name="dataElementCount">count of strings</param>
    template <typename OP>
    static void
    ColCol(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, GPUMemory::GPUString BCol, int32_t dataElementCount)
    {
        GPUStringBinary::Run<OP>(output, ACol, true, BCol, true, dataElementCount);
    }

    /// Binary string operation string column - string constant, output is string column
    /// <param name="output">output string column</param>
    /// <param name="ACol">first input string column (GPUString)</param>
    /// <param name="BConst">second input string constant (GPUString)</param>
    /// <param name="dataElementCount">count of strings</param>
    template <typename OP>
    static void
    ColConst(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, GPUMemory::GPUString BConst, int32_t dataElementCount)
    {
        GPUStringBinary::Run<OP>(output, ACol, true, BConst, false, dataElementCount);
    }

    /// Binary string operation string constant - string column, output is string column
    /// <param name="output">output string column</param>
    /// <param name="AConst">first input string constant (GPUString)</param>
    /// <param name="BCol">second input string column (GPUString)</param>
    /// <param name="dataElementCount">count of strings</param>
    template <typename OP>
    static void
    ConstCol(GPUMemory::GPUString& output, GPUMemory::GPUString AConst, GPUMemory::GPUString BCol, int32_t dataElementCount)
    {
        GPUStringBinary::Run<OP>(output, AConst, false, BCol, true, dataElementCount);
    }

    /// Binary string operation string constant - string constant, output is string constant
    /// <param name="output">output string constant (GPUString column with 1 constant)</param>
    /// <param name="AConst">first input string constant (GPUString)</param>
    /// <param name="BCol">second input string constant (GPUString)</param>
    /// <param name="dataElementCount">count of strings</param>
    template <typename OP>
    static void
    ConstConst(GPUMemory::GPUString& output, GPUMemory::GPUString AConst, GPUMemory::GPUString BConst, int32_t dataElementCount)
    {
        GPUStringBinary::Run<OP>(output, AConst, true, BConst, true, dataElementCount);
        // TODO expand?
    }
};
