#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "ArithmeticUnaryOperations.h"
#include "GPUStringUnary.cuh"
#include "DateOperations.h"
#include "GPUDate.cuh"
#include "GPUCast.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic unary operation with column and column
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template <typename OP, typename T, typename U>
__global__ void kernel_arithmetic_unary(T* output, U ACol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] = OP{}.template operator()<T, typename std::remove_pointer<U>::type>(maybe_deref(ACol, i));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for unary arithmetic functions
template <typename OP, typename T, typename U, class Enable = void>
class GPUArithmeticUnary
{
public:
    /// Arithmetic unary operation with values from column
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void ArithmeticUnary(T* output, U ACol, int32_t dataElementCount)
    {
        kernel_arithmetic_unary<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, ACol, dataElementCount);
    }
};

template <typename OP, typename T, typename U>
class GPUArithmeticUnary<OP,
                         T,
                         U,
                         typename std::enable_if<std::is_same<typename std::remove_pointer<T>::type, std::string>::value &&
                                                 std::is_same<typename std::remove_pointer<U>::type, std::string>::value>::type>
{
public:
    /// String unary operations which return string, for column
    /// <param name="output">output string column</param>
    /// <param name="inCol">input string column (GPUString)</param>
    /// <param name="dataElementCount">input string count</param>
    static void ArithmeticUnary(GPUMemory::GPUString& output, GPUMemory::GPUString inCol, int32_t dataElementCount)
    {
        output = OP{}(inCol, dataElementCount);
    }
};

template <typename OP, typename T, typename U>
class GPUArithmeticUnary<OP,
                         T,
                         U,
                         typename std::enable_if<std::is_same<typename std::remove_pointer<T>::type, int32_t>::value &&
                                                 std::is_same<typename std::remove_pointer<U>::type, std::string>::value>::type>
{
public:
    /// String unary operations which return string, for column
    /// <param name="output">output string column</param>
    /// <param name="inCol">input string column (GPUString)</param>
    /// <param name="dataElementCount">input string count</param>
    static void ArithmeticUnary(int32_t* outCol, GPUMemory::GPUString inCol, int32_t dataElementCount)
    {
        if constexpr (std::is_pointer<U>::value)
        {
            Context& context = Context::getInstance();
            kernel_lengths_from_indices<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                outCol, inCol.stringIndices, dataElementCount);
            CheckCudaError(cudaGetLastError());
        }
        else
        {
            // Copy single index to host
            int64_t hostIndex;
            GPUMemory::copyDeviceToHost(&hostIndex, inCol.stringIndices, 1);
            // Cast to int32 and copy to return result
            int32_t length = static_cast<int32_t>(hostIndex);
            GPUMemory::copyHostToDevice(outCol, &length, 1);
        }
    }
};

template <typename OP, typename T, typename U>
class GPUArithmeticUnary<OP,
                         T,
                         U,
                         typename std::enable_if<std::is_same<OP, DateOperations::toString>::value &&
                                                 std::is_same<typename std::remove_pointer<T>::type, std::string>::value &&
                                                 std::is_same<typename std::remove_pointer<U>::type, int64_t>::value>::type>
{
public:
    static void ArithmeticUnary(GPUMemory::GPUString& output, U dateTimeCol, int32_t dataElementCount)
    {
        static_assert(std::is_same<typename std::remove_pointer<U>::type, int64_t>::value,
                      "DateTime can only be extracted from int64 columns");

        if (dataElementCount > 0)
        {

            // Length of date format "YYYY-MM-DD HH:MM:SS" = 19

            const int32_t dateFormatLength = 19;

            GPUMemory::alloc(&(output.stringIndices), dataElementCount);

            cuda_ptr<int32_t> lengths(dataElementCount);
            kernel_fill_array<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(lengths.get(), dateFormatLength,
                                                                        dataElementCount);

            GPUReconstruct::PrefixSum(output.stringIndices, lengths.get(), dataElementCount);

            int64_t totalCharCount;
            GPUMemory::copyDeviceToHost(&totalCharCount, output.stringIndices + dataElementCount - 1, 1);
            GPUMemory::alloc(&(output.allChars), totalCharCount);

            cuda_ptr<int32_t> years(dataElementCount);
            cuda_ptr<int32_t> months(dataElementCount);
            cuda_ptr<int32_t> days(dataElementCount);
            cuda_ptr<int32_t> hours(dataElementCount);
            cuda_ptr<int32_t> minutes(dataElementCount);
            cuda_ptr<int32_t> seconds(dataElementCount);


            kernel_arithmetic_unary<DateOperations::year, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    years.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::month, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    months.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::day, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    days.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::hour, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    hours.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::minute, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    minutes.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::second, int32_t, U>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    seconds.get(), dateTimeCol, dataElementCount);

            kernel_fill_date_string<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDim()>>>(output, years.get(),
                                                                              months.get(), days.get(),
                                                                              hours.get(), minutes.get(),
                                                                              seconds.get(), dataElementCount);
            cudaDeviceSynchronize();
            CheckCudaError(cudaGetLastError());
        }
        else
        {
            output.allChars = nullptr;
            output.allChars = nullptr;
        }
    }
};

template <typename OP, typename T, typename U>
class GPUArithmeticUnary<OP,
                         T,
                         U,
                         typename std::enable_if<std::is_same<OP, CastOperations::ToString>::value &&
                                                 std::is_same<typename std::remove_pointer<T>::type, std::string>::value &&
                                                 std::is_arithmetic<typename std::remove_pointer<U>::type>::value>::type>
{
public:
    /// String unary operations which return string, for column
    /// <param name="output">output string column</param>
    /// <param name="inCol">input string column (GPUString)</param>
    /// <param name="dataElementCount">input string count</param>
    static void ArithmeticUnary(GPUMemory::GPUString& outCol, U inCol, int32_t dataElementCount)
    {
        GPUCast::CastNumericToString(outCol, inCol, dataElementCount);
    }
};
