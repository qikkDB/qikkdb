#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "../../Types/Point.pb.h"
#include "../../Types/ComplexPolygon.pb.h"
#include "cuda_ptr.h"

#include "../../../cub/cub.cuh"

/// Precision of generated WKT floats as number of decimal places
/// (4 is for about 10 m accuracy, 3 for 100 m)
__device__ const int32_t WKT_DECIMAL_PLACES = 4;

/// POLYGON word
__device__ const char WKT_POLYGON[] = "POLYGON";

/// Kernel for reconstructing buffer according to calculated prefixSum and inMask
template<typename T>
__global__ void kernel_reconstruct_col(T *outData, T *ACol, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount)
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
}

/// Kernel for generating array with sorted indexes which point to values where mask is 1.
template<typename T>
__global__ void kernel_generate_indexes(T *outData, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount)
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
}

/// Kernel for mask expanding in order to reconstruct sub-polygons (pointIdx and pointCount arrays).
/// Expanding is performed by propagating values from inMask based on counts.
__global__ void kernel_generate_submask(int8_t *outMask, int8_t *inMask, int32_t *polyIdx, int32_t *polyCount, int32_t polyIdxSize);

/// Kernel for predicitng lenghts of WKT strings based on GPUPolygon struct
__global__ void kernel_predict_wkt_lengths(int32_t * outStringLengths, GPUMemory::GPUPolygon inPolygon, int32_t dataElementCount);

/// Kernel for convertion of GPUPolygon representation to WKT representation (GPUString)
__global__ void kernel_convert_poly_to_wkt(GPUMemory::GPUString outWkt, GPUMemory::GPUPolygon inPolygon, int32_t dataElementCount);

/// Class for reconstructing buffers according to mask
class GPUReconstruct {
private:
	/// Calculate count of elements in subarray (for GPUPolygon struct arrays)
	/// based on last number in indices and counts
	static int32_t CalculateCount(int32_t * indices, int32_t * counts, int32_t size);

public:

	/// Reconstruct block of column and copy result to host (CPU)
	/// <param name="outCol">CPU buffer which will be filled with result</param>
	/// <param name="outDataElementCount">CPU pointer, will be filled with one number representing reconstructed rows in block</param>
	/// <param name="ACol">input block</param>
	/// <param name="inMask">input mask</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename T>
	static void reconstructCol(T *outData, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount)
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
		CheckCudaError(cudaGetLastError());
	}

	/// Reconstruct block of column and keep reuslt on GPU
	/// <param name="outCol">will be allocated on GPU and filled with result</param>
	/// <param name="outDataElementCount">CPU pointer, will be filled with one number representing reconstructed rows in block</param>
	/// <param name="ACol">input block</param>
	/// <param name="inMask">input mask</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename T>
	static void reconstructColKeep(T **outCol, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		try
		{
			if (inMask)		// If inMask is not nullptr
			{
				// Malloc a new buffer for the prefix sum vector
				cuda_ptr<int32_t> prefixSumPointer(dataElementCount);

				PrefixSum(prefixSumPointer.get(), inMask, dataElementCount);
				GPUMemory::copyDeviceToHost(outDataElementCount, prefixSumPointer.get() + dataElementCount - 1, 1);
				if(*outDataElementCount > 0)
				{ 
					GPUMemory::alloc<T>(outCol, *outDataElementCount);
					// Construct the output based on the prefix sum
					kernel_reconstruct_col << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
							(*outCol, ACol, prefixSumPointer.get(), inMask, dataElementCount);
				}
				else
				{
					*outCol = nullptr;
				}
			}
			else if (*outCol != ACol)	// If inMask is nullptr, just copy whole ACol to outCol (if they are not pointing to the same blocks)
			{
				GPUMemory::alloc<T>(outCol, dataElementCount);
				GPUMemory::copyDeviceToDevice(*outCol, ACol, dataElementCount);
				*outDataElementCount = dataElementCount;
			}
		}
		catch (...)
		{
			if (outCol)
			{
				GPUMemory::free(outCol);
			}
			throw;
		}

		// Get last error
		CheckCudaError(cudaGetLastError());
	}

	/// Reconstruct GPUString column and copy to CPU memory
	/// <param name="outStringData">output CPU string array</param>
	/// <param name="outDataElementCount">reconstructed data elemen (string) count</param>
	/// <param name="inStringCol">input GPUString column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (string) count</param>
	static void ReconstructStringCol(std::string *outStringData, int32_t *outDataElementCount,
		GPUMemory::GPUString inStringCol, int8_t *inMask, int32_t inDataElementCount);

	/// Convert polygons to WKTs (GPUPolygon column to GPUString columns)
	/// <param name="outStringCol">output GPUString column</param>
	/// <param name="inPolygonCol">input GPUPolygon column</param>
	/// <param name="dataElementCount">input data element (complex polygon) count</param>
	static void ConvertPolyColToWKTCol(GPUMemory::GPUString *outStringCol,
		GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount);

	/// Recontruct GPUPolygon column and keep on GPU in the same format
	/// <param name="outCol">output GPUPolygon column</param>
	/// <param name="outDataElementCount">reconstructed data element (complex polygon) count</param>
	/// <param name="inCol">input GPUPolygon column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (complex polygon) count</param>
	static void ReconstructPolyColKeep(GPUMemory::GPUPolygon *outCol, int32_t *outDataElementCount,
		GPUMemory::GPUPolygon inCol, int8_t *inMask, int32_t inDataElementCount);

	/// Reconstruct GPUPolygon column and convert to WKT string array on CPU
	/// <param name="outStringData">output CPU string array</param>
	/// <param name="outDataElementCount">reconstructed data element (WKT string) count</param>
	/// <param name="inPolygonCol">input GPUPolygon column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (complex polygon) count</param>
	static void ReconstructPolyColToWKT(std::string * outStringData, int32_t *outDataElementCount,
		GPUMemory::GPUPolygon inPolygonCol, int8_t *inMask, int32_t inDataElementCount);

	/// Function for generating array with sorted indexes which point to values where mask is 1.
	/// Result is copied to host.
	/// <param name="outData">result array (must be allocated on host)</param>
	/// <param name="outDataElementCount">result data element count</param>
	/// <param name="inMask">input mask to process (on device)</param>
	/// <param name="dataElementCount">input data element count</param>
	template<typename T, typename M>
	static void GenerateIndexes(T *outData, int32_t *outDataElementCount, M *inMask, int32_t dataElementCount)
	{
		if (dataElementCount > 0)
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
		else
		{
			*outDataElementCount = 0;
		}
	}

	/// Function for generating array with sorted indexes which point to values where mask is 1.
	/// Result is kept on device.
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
			try
			{
				// A new buffer for the prefix sum vector
				cuda_ptr<int32_t> prefixSumPointer(dataElementCount);
			
				// Run prefix sum
				PrefixSum(prefixSumPointer.get(), inMask, dataElementCount);
			
				// Copy the output size to host
				GPUMemory::copyDeviceToHost(outDataElementCount, prefixSumPointer.get() + dataElementCount - 1, 1);
				if (*outDataElementCount > 0)
				{
					// Allocate array for outData with needed size
					GPUMemory::alloc<T>(outData, *outDataElementCount);

					// Call kernel for generating indexes
					kernel_generate_indexes << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
						(*outData, prefixSumPointer.get(), inMask, dataElementCount);
				}
				else
				{
					*outData = nullptr;
				}
			}
			catch (...)
			{
				if(outData)
				{
					GPUMemory::free(outData);
				}
				throw;
			}
		}
		else  // Version without mask is not supported in GenerateIndexes
		{
			CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR, "inMask cannot be nullptr in GenerateIndexes");
		}

		// Get last error
		CheckCudaError(cudaGetLastError());
	}

	/// Calculate just prefix sum from input mask (keep result on GPU)
	/// <param name="prefixSumPointer">output GPU buffer which will be filled with result</param>
	/// <param name="inMask">input mask</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename T, typename M>
	static void PrefixSum(T* prefixSumPointer, M* inMask, int32_t dataElementCount)
	{
		// Start the collumn reconstruction
		size_t tempBufferSize = 0;
		// Calculate the prefix sum for getting tempBufferSize (in-place scan)
		cub::DeviceScan::InclusiveSum(nullptr, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
		// Temporary storage
		cuda_ptr<int8_t>tempBuffer(tempBufferSize);
		// Run inclusive prefix sum
		cub::DeviceScan::InclusiveSum(tempBuffer.get(), tempBufferSize, inMask, prefixSumPointer, dataElementCount);
	}


	/// Calculate exclusive prefix sum from input mask (keep result on GPU)
	/// <param name="prefixSumPointer">output GPU buffer which will be filled with result</param>
	/// <param name="inMask">input mask</param>
	/// <param name="dataElementCount">data element count in the inMask</param>
	template<typename M>
	static void PrefixSumExclusive(int32_t* prefixSumPointer, M* inMask, int32_t dataElementCount)
	{
		// Start the collumn reconstruction
		size_t tempBufferSize = 0;
		// Calculate the prefix sum for getting tempBufferSize (in-place scan)
		cub::DeviceScan::ExclusiveSum(nullptr, tempBufferSize, inMask, prefixSumPointer, dataElementCount);
		// Temporary storage
		cuda_ptr<int8_t>tempBuffer(tempBufferSize);
		// Run exclusive prefix sum
		cub::DeviceScan::ExclusiveSum(tempBuffer.get(), tempBufferSize, inMask, prefixSumPointer, dataElementCount);
	}

};

/// Specialization for Point (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructCol<ColmnarDB::Types::Point>(ColmnarDB::Types::Point *outData,
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount);

/// Specialization for ComplexPolygon (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructCol<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon *outData,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount);


/// Specialization for Point (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::Point>(ColmnarDB::Types::Point **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount);

/// Specialization for ComplexPolygon (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount);
