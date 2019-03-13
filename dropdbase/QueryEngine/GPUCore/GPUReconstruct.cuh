#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "GPUMemory.cuh"

#include "../../../cub/cub.cuh"

template<typename T>
__global__ void kernel_reconstruct_col(T *outData, int32_t *outDataElementCount, T *ACol, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount)
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

	// Fetch the size of the output - the last item of the inclusive prefix sum // TODO delete, no needed anymore
	if (idx == 0)
	{
		outDataElementCount[0] = prefixSum[dataElementCount - 1];
	}
}

// Kernel for generating array with sorted indexes which point to values where mask is 1.
template<typename T>
__global__ void kernel_generate_indexes(T *outData, int32_t *outDataElementCount, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount)
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
			outData[prefixSum[i] - 1] = i;
		}
	}

	// Fetch the size of the output - the last item of the inclusive prefix sum // TODO delete, no needed anymore
	if (idx == 0)
	{
		outDataElementCount[0] = prefixSum[dataElementCount - 1];
	}
}

class GPUReconstruct {
public:

	template<typename T, typename M>
	static void reconstructCol(T *outData, int32_t *outDataElementCount, T *ACol, M *inMask, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		if (inMask)		// If inMask is not nullptr
		{
			// Malloc a new buffer for the output vector -GPU side
			T *outDataGPUPointer = nullptr;

			// Call reconstruct col keep
			reconstructColKeep(&outDataGPUPointer, outDataElementCount, ACol, inMask, dataElementCount);

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


	template<typename T, typename M>
	static void reconstructColKeep(T **outCol, int32_t *outDataElementCount, T *ACol, M *inMask, int32_t dataElementCount)
    {
        static_assert(std::is_same<M, int8_t>::value || std::is_same<M, int32_t>::value,
                      "ReconstructCol: inMask have to be int8_t* or int32_t*");
		Context& context = Context::getInstance();

		if (inMask)		// If inMask is not nullptr
		{
			// Malloc a new buffer for the prefix sum vector
			int32_t* prefixSumPointer = nullptr;
			GPUMemory::alloc(&prefixSumPointer, dataElementCount);

			// Malloc a new buffer for the output size
			int32_t* outDataElementCountPointer = nullptr;
			GPUMemory::alloc(&outDataElementCountPointer, 1);

			PrefixSum(prefixSumPointer, inMask, dataElementCount);
			GPUMemory::copyDeviceToHost(outDataElementCount, prefixSumPointer + dataElementCount - 1, 1);
			GPUMemory::alloc<T>(outCol, *outDataElementCount);
			// Construct the output based on the prefix sum
			kernel_reconstruct_col << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(*outCol, outDataElementCountPointer, ACol, prefixSumPointer, inMask, dataElementCount);

			// Free the memory
			GPUMemory::free(prefixSumPointer);
			GPUMemory::free(outDataElementCountPointer);
		}
		else if (*outCol != ACol)	// If inMask is nullptr, just copy whole ACol to outCol (if they are not pointing to the same blocks)
		{
			GPUMemory::alloc<T>(outCol, dataElementCount);
			GPUMemory::copyDeviceToDevice(*outCol, ACol, dataElementCount);
			*outDataElementCount = dataElementCount;
		}

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}



	/// <summary>
	/// Function for generating array with sorted indexes which point to values where mask is 1.
	/// Result is copied to host.
	/// </summary>
	/// <param name="outData">result array (must be allocated on host)</param>
	/// <param name="outDataElementCount">result data element count</param>
	/// <param name="inMask">input mask to process (on device)</param>
	/// <param name="dataElementCount">input data element count</param>
	template<typename T, typename M>
	static void GenerateIndexes(T *outData, int32_t *outDataElementCount, M *inMask, int32_t dataElementCount)
	{
		// New buffer for the output vector - GPU side
		T *outDataGPUPointer = nullptr;

		// Call keep version
		GenerateIndexesKeep(&outDataGPUPointer, outDataElementCount, inMask, dataElementCount);

		// Copy the generated output from GPU (device) to host
		GPUMemory::copyDeviceToHost(outData, outDataGPUPointer, *outDataElementCount);

		// Free the memory
		GPUMemory::free(outDataGPUPointer);
	}

	/// <summary>
	/// Function for generating array with sorted indexes which point to values where mask is 1.
	/// Result is keeped on device.
	/// </summary>
	/// <param name="outData">pointer to result array (this function also allocates it)</param>
	/// <param name="outDataElementCount">result data element count</param>
	/// <param name="inMask">input mask to process (on device)</param>
	/// <param name="dataElementCount">input data element count</param>
	template<typename T, typename M>
	static void GenerateIndexesKeep(T **outData, int32_t *outDataElementCount, M *inMask, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		if (inMask)
		{
			// Malloc a new buffer for the prefix sum vector
			int32_t* prefixSumPointer = nullptr;
			GPUMemory::alloc(&prefixSumPointer, dataElementCount);

			// Malloc a new buffer for the output size
			int32_t* outDataElementCountPointer = nullptr;
			GPUMemory::alloc(&outDataElementCountPointer, 1);
			
			// Run prefix sum
			PrefixSum(prefixSumPointer, inMask, dataElementCount);
			
			// Copy the output size to host
			GPUMemory::copyDeviceToHost(outDataElementCount, prefixSumPointer + dataElementCount - 1, 1);

			// Allocate array for outData with needed size
			GPUMemory::alloc<T>(outData, *outDataElementCount);

			// Call kernel for generating indexes
			kernel_generate_indexes << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(*outData, outDataElementCountPointer, prefixSumPointer, inMask, dataElementCount);

			// Free the memory
			GPUMemory::free(prefixSumPointer);
			GPUMemory::free(outDataElementCountPointer);
		}
		else  // Version without mask is not supported in GenerateIndexes
		{
			QueryEngineError::setType(QueryEngineError::GPU_EXTENSION_ERROR, "inMask cannot be nullptr in GenerateIndexes");
		}

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename M>
	static void PrefixSum(int32_t* prefixSumPointer, M* inMask, int32_t dataElementCount)
	{
		// Start the collumn reconstruction
		void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;
		// Calculate the prefix sum
		// in-place scan
		cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
		// Allocate temporary storage
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
		// Run inclusive prefix sum
		cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
		GPUMemory::free(tempBuffer);
	}

};


