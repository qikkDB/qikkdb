#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"

template <typename T>
__global__ void
kernel_predict_length_cut(int32_t* newLengths, GPUMemory::GPUString inCol, int32_t stringCount, T lengthLimit, int32_t numberCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    const bool stringCol = (stringCount > 1);
    for (int32_t i = idx; i < stringCol ? stringCount : numberCount; i += stride)
    {
        const int32_t stringI = stringCol ? i : 0;
        const int64_t index = (stringI == 0) ? 0 : inCol.stringIndices[stringI - 1];
        const int32_t length = static_cast<int32_t>(inCol.stringIndices[stringI] - index);
        newLengths[i] = min(length, static_cast<int32_t>(maybe_deref(lengthLimit, i)));
    }
}


template <typename OP>
__global__ void
kernel_string_cut(GPUMemory::GPUString outCol, int32_t* newLengths, int32_t outStringCount, GPUMemory::GPUString inCol, int32_t inStringCount)
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
            outCol.allChars[outCol.stringIndices[outI] + j] =
                inCol.allChars[inCol.stringIndices[inI] + j + OP{}(newLengths[outI], outCol.stringIndices, inI)];
        }
    }
}

namespace StringBinaryOperations
{
	struct left
	{
		template <typename T>
		__device__ int32_t operator()(int32_t newLength, int64_t* oldIndices, int32_t i) const
		{
			return 0;
		}
	};

	struct right
	{
		template <typename T>
		__device__ int32_t operator()(int32_t newLength, int32_t* oldIndices, int32_t i) const
		{
			return static_cast<int32_t>(oldIndices[i] - ((i == 0) ? 0 : oldIndices[i - 1])) - newLength;
		}
	};
} // namespace StringBinaryOperations


class GPUStringBinary
{
private:
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
			throw std::runtime_error("String and number count must be the same for col-col operations");
		}

		Context &context = Context::getInstance();
		int32_t outputCount = max(stringCount, numberCount);

		// Predict new lengths
		cuda_ptr<int32_t> newLenghts(outputCount);
		kernel_predict_length_cut << < context.calcGridDim(outputCount), context.getBlockDim() >> >
			(newLenghts.get(), ACol, stringCount, BCol, numberCount);

		// Alloc and compute new stringIndices
		GPUMemory::alloc(&(outCol.stringIndices), outputCount);
		PrefixSum(outCol.stringIndices, newLenghts.get(), outputCount);

		// Get total char count and alloc allChars
		int64_t outTotalCharCount;
		GPUMemory::copyDeviceToHost(&outTotalCharCount, outCol.stringIndices + outputCount - 1, 1);
		GPUMemory::alloc(&(outCol.allChars), outTotalCharCount);

		// Copy appropriate part of strings
		kernel_string_cut<OP> << <context.calcGridDim(outputCount), context.getBlockDim() >> >
			(outCol, newLenghts.get(), outputCount, ACol, stringCount);
		
		CheckCudaError(cudaGetLastError());
    }

public:
    template <typename OP, typename T>
    static void ColCol(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, T BCol, int32_t dataElementCount)
    {
		GPUStringBinary::Run(output, ACol, dataElementCount, BCol, dataElementCount);
    }

    template <typename OP, typename T>
    static void ColConst(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, T BConst, int32_t dataElementCount)
    {
		GPUStringBinary::Run(output, ACol, dataElementCount, BConst, 1);
    }

    template <typename OP, typename T>
    static void ConstCol(GPUMemory::GPUString& output, GPUMemory::GPUString AConst, T BCol, int32_t dataElementCount)
    {
		GPUStringBinary::Run(output, AConst, 1, BCol, dataElementCount);
    }

    template <typename OP, typename T>
    static void ConstConst(GPUMemory::GPUString& output, GPUMemory::GPUString AConst, T BConst, int32_t dataElementCount)
    {
		GPUStringBinary::Run(output, AConst, 1, BConst, 1);
		// TODO expand?
    }
};
