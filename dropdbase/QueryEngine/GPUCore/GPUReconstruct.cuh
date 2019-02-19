#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUTypeWidthManip.cuh"

#include "../../../cub/cub.cuh"

template<typename T>
__global__ void kernel_reconstruct_col(T *outData, int32_t *outDataElementCount, T *ACol, int32_t *prefixSum, int32_t *inMask, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// Select the elemnts that are "visible" in the mask
		// If the mask is 1 for the output, use the prefix sum for array compaction
		// The prefix sum includes values from the input array on the same element so the index has to be modified
		if (inMask[i] && (prefixSum[i] - 1) >= 0)
		{
			outData[prefixSum[i] - 1] = ACol[i];
		}
	}

	// Fetch the size of the output - the last item of the inclusive prefix sum
	if (idx == 0) 
	{
		outDataElementCount[0] = prefixSum[dataElementCount - 1];
	}
}

class GPUReconstruct {
public:
	template<typename T>
	static void reconstructCol(T *outData, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount)
	{
		if (inMask)		// If inMask is not nullptr
		{
			// Malloc a new buffer for the output vector -GPU side
			T *outDataGPUPointer = nullptr;
			GPUMemory::alloc(&outDataGPUPointer, dataElementCount);

			// Call reconstruct col keep
			reconstructColKeep(outDataGPUPointer, outDataElementCount, ACol, inMask, dataElementCount);

			// Copy the generated output back from the GPU
			GPUMemory::copyDeviceToHost(outData, outDataGPUPointer, *outDataElementCount);

			// Free the memory
			GPUMemory::free(outDataGPUPointer);
		}
		else		// If inMask is nullptr, just copy whole ACol to outData
		{
			GPUMemory::copyDeviceToHost(outData, ACol, dataElementCount);
			*outDataElementCount = dataElementCount;
		}

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename T>
	static void reconstructColKeep(T *outCol, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		if (inMask)		// If inMask is not nullptr
		{
			// Typecasted array of bool elements of the inMask
			int32_t* inMask32Pointer = nullptr;
			GPUMemory::alloc(&inMask32Pointer, dataElementCount);

			// Malloc a new buffer for the prefix sum vector
			int32_t* prefixSumPointer = nullptr;
			GPUMemory::alloc(&prefixSumPointer, dataElementCount);

			// Malloc a new buffer for the output size
			int32_t* outDataElementCountPointer = nullptr;
			GPUMemory::alloc(&outDataElementCountPointer, 1);

			// Typecast the input mask array
			GPUTypeWidthManip::convertBuffer(inMask32Pointer, inMask, dataElementCount);

			// Start the collumn reconstruction
			// Calculate the prefix sum
			// in-place scan
			void* tempBuffer = nullptr;
			size_t tempBufferSize = 0;
			cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask32Pointer, prefixSumPointer, dataElementCount);
			// Allocate temporary storage
			GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
			// Run inclusive prefix sum
			cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask32Pointer, prefixSumPointer, dataElementCount);
			GPUMemory::free(tempBuffer);
			// Construct the output based on the prefix sum
			kernel_reconstruct_col << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(outCol, outDataElementCountPointer, ACol, prefixSumPointer, inMask32Pointer, dataElementCount);
			cudaDeviceSynchronize();

			// Copy the generated output back from the GPU
			GPUMemory::copyDeviceToHost(outDataElementCount, outDataElementCountPointer, 1);

			// Free the memory
			GPUMemory::free(inMask32Pointer);
			GPUMemory::free(prefixSumPointer);
			GPUMemory::free(outDataElementCountPointer);
		}
		else if (outCol != ACol)	// If inMask is nullptr, just copy whole ACol to outCol (if they are not pointing to the same blocks)
		{
			GPUMemory::copyDeviceToDevice(outCol, ACol, dataElementCount);
			*outDataElementCount = dataElementCount;
		}

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename T>
	static void reconstructColKeep(T *outCol, int32_t *outDataElementCount, T *ACol, int32_t *inMask, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		if (inMask)		// If inMask is not nullptr
		{
			// Malloc a new buffer for the prefix sum vector
			int32_t* prefixSumPointer = nullptr;
			GPUMemory::alloc(&prefixSumPointer, dataElementCount);

			// Malloc a new buffer for the output size
			int32_t* outSizePointer = nullptr;
			GPUMemory::alloc(&outSizePointer, 1);

			// Start the collumn reconstruction
			// Calculate the prefix sum
			// in-place scan
			void* tempBuffer = nullptr;
			size_t tempBufferSize = 0;
			cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
			// Allocate temporary storage
			GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
			// Run inclusive prefix sum
			cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
			GPUMemory::free(tempBuffer);
			// Construct the output based on the prefix sum
			kernel_reconstruct_col << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(outCol, outSizePointer, ACol, prefixSumPointer, inMask, dataElementCount);
			cudaDeviceSynchronize();

			// Copy the generated output back from the GPU
			GPUMemory::copyDeviceToHost(outDataElementCount, outSizePointer, 1);

			// Free the memory
			GPUMemory::free(prefixSumPointer);
			GPUMemory::free(outSizePointer);
		}
		else if (outCol != ACol)	// If inMask is nullptr, just copy whole ACol to outCol (if they are not pointing to the same blocks)
		{
			GPUMemory::copyDeviceToDevice(outCol, ACol, dataElementCount);
			*outDataElementCount = dataElementCount;
		}

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}
};


