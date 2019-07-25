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

/// POINT word
__device__ const char WKT_POINT[] = "POINT";

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

/// Kernel for reconstructing null masks according to calculated prefixSum and inMask
__global__ void kernel_reconstruct_null_mask(int32_t *outData, int8_t *ACol, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount);

/// Kernel for compressing null masks from memory-wasting int8_t* array
__global__ void kernel_compress_null_mask(int32_t *outData, int8_t *ACol, int32_t dataElementCount);

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

/// Kernel for calculating lengths from indices (difference of indices)
/// - reversed inclusive prefix sum.
template<typename L, typename I>
__global__ void kernel_lengths_from_indices(L * outLengths, I * inIndices, int32_t dataElementCount)
{

	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		if (i == 0)
		{
			outLengths[i] = inIndices[i];
		}
		else
		{
			outLengths[i] = inIndices[i] - inIndices[i - 1];
		}
	}
}

/// Kernel for reconstruction of chars in GPUString
__global__ void kernel_reconstruct_string_chars(GPUMemory::GPUString outStringCol,
	GPUMemory::GPUString inStringCol, int32_t * inStringLengths,
	int32_t *prefixSum, int8_t *inMask, int32_t stringCount);

/// Kernel for mask expanding in order to reconstruct sub-polygons (pointIdx and pointCount arrays).
/// Expanding is performed by propagating values from inMask based on counts.
__global__ void kernel_generate_poly_submask(int8_t *outMask, int8_t *inMask, GPUMemory::GPUPolygon polygon, int32_t size);

__global__ void kernel_generate_point_submask(int8_t *outMask, int8_t *inMask, GPUMemory::GPUPolygon polygon, int32_t size);

/// Kernel for predicitng lenghts of WKT strings based on GPUPolygon struct
__global__ void kernel_predict_wkt_lengths(int32_t * outStringLengths, GPUMemory::GPUPolygon inPolygon, int32_t dataElementCount);

/// Kernel for reconstructing polygon subPolygons
__global__ void kernel_reconstruct_polyCount_col(int32_t * outPolyCount, GPUMemory::GPUPolygon inPolygon, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount);

/// Kernel for reconstructing polygon points
__global__ void kernel_reconstruct_pointCount_col(int32_t *outPointCount, GPUMemory::GPUPolygon inPolygon, int32_t *prefixSum, int8_t *inMask, int32_t dataElementCount);

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
	static void reconstructCol(T *outData, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount, int8_t* outNullMask = nullptr, int8_t* nullMask = nullptr)
	{
		Context& context = Context::getInstance();

		if (inMask)
		{
			// Malloc a new buffer for the output vector -GPU side
			T *outDataGPUPointer = nullptr;
			int8_t *outNullMaskGPUPointer = nullptr;
			try
			{
				// Call reconstruct col keep
				reconstructColKeep(&outDataGPUPointer, outDataElementCount, ACol, inMask, dataElementCount, &outNullMaskGPUPointer, nullMask);

				// Copy the generated output back from the GPU
				GPUMemory::copyDeviceToHost(outData, outDataGPUPointer, *outDataElementCount);
				if(outNullMaskGPUPointer)
				{
					size_t outBitMaskSize = (*outDataElementCount + sizeof(char)*8 - 1) / (sizeof(char)*8);
					GPUMemory::copyDeviceToHost(outNullMask, outNullMaskGPUPointer, outBitMaskSize);
					GPUMemory::free(outNullMaskGPUPointer);
				}
				// Free the memory
				if(outDataGPUPointer)
				{
					GPUMemory::free(outDataGPUPointer);
				}
			}
			catch(...)
			{
				if(outNullMaskGPUPointer)
				{
					GPUMemory::free(outNullMaskGPUPointer);
				}
				if(outDataGPUPointer)
				{
					GPUMemory::free(outDataGPUPointer);
				}
				throw;
			}

		}
		else
		{
			GPUMemory::copyDeviceToHost(outData, ACol, dataElementCount);
			*outDataElementCount = dataElementCount;
			if(nullMask)
			{
				size_t outBitMaskSize = (*outDataElementCount + sizeof(char)*8 - 1) / (sizeof(char)*8);
				GPUMemory::copyDeviceToHost(outNullMask, nullMask, outBitMaskSize);
			}
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
	static void reconstructColKeep(T **outCol, int32_t *outDataElementCount, T *ACol, int8_t *inMask, int32_t dataElementCount, int8_t** outNullMask = nullptr, int8_t* nullMask = nullptr)
	{
		Context& context = Context::getInstance();

		try
		{
			if (dataElementCount > 0)
			{
				if(inMask)
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
						if(nullMask)
						{
							size_t outBitMaskSize = (*outDataElementCount + sizeof(int32_t)*8 - 1) / (sizeof(int32_t)*8);
							GPUMemory::allocAndSet(outNullMask, 0, outBitMaskSize);
							kernel_reconstruct_null_mask << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
								(reinterpret_cast<int32_t*>(*outNullMask), nullMask, prefixSumPointer.get(), inMask, dataElementCount);
						}
					}
					else
					{
						*outCol = nullptr;
						if (outNullMask)
						{
							*outNullMask = nullptr;
						}
					}
					
				}
				else if (*outCol != ACol)	// If inMask is nullptr, just copy whole ACol to outCol (if they are not pointing to the same blocks)
				{
					GPUMemory::alloc<T>(outCol, dataElementCount);
					GPUMemory::copyDeviceToDevice(*outCol, ACol, dataElementCount);
					size_t outBitMaskSize = (dataElementCount + sizeof(char)*8 - 1) / (sizeof(char)*8);
					if(nullMask)
					{
						GPUMemory::alloc(outNullMask, outBitMaskSize);
						GPUMemory::copyDeviceToDevice(*outNullMask, nullMask, outBitMaskSize);
					}
					*outDataElementCount = dataElementCount;
				}
			}
			else
			{
				*outCol = nullptr;
				*outDataElementCount = 0;
				*outNullMask = nullptr;
			}
		}
		catch (...)
		{
			if (*outCol)
			{
				GPUMemory::free(*outCol);
			}
			if (*outNullMask)
			{
				GPUMemory::free(*outNullMask);
			}
			throw;
		}

		// Get last error
		CheckCudaError(cudaGetLastError());
	}

	/// Reconstruct GPUString column (keep result on GPU)
	/// <param name="outStringCol">output GPUString column</param>
	/// <param name="outDataElementCount">reconstructed data element (string) count</param>
	/// <param name="inStringCol">input GPUString column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (string) count</param>
	static void ReconstructStringColKeep(GPUMemory::GPUString *outStringCol, int32_t *outDataElementCount,
		GPUMemory::GPUString inStringCol, int8_t *inMask, int32_t inDataElementCount, int8_t** outNullMask = nullptr, int8_t* nullMask = nullptr);

	/// Reconstruct GPUString column and copy to CPU memory
	/// <param name="outStringData">output CPU string array</param>
	/// <param name="outDataElementCount">reconstructed data element (string) count</param>
	/// <param name="inStringCol">input GPUString column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (string) count</param>
	static void ReconstructStringCol(std::string *outStringData, int32_t *outDataElementCount,
		GPUMemory::GPUString inStringCol, int8_t *inMask, int32_t inDataElementCount, int8_t* outNullMask = nullptr, int8_t* nullMask = nullptr);

	/// Reconstruct GPUString column to two arrays: string lengths and all chars
	/// and copy them to the CPU.
	static void ReconstructStringColRaw(std::vector<int32_t>& keysStringLengths, std::vector<char>& keysAllChars,
		int32_t *outDataElementCount, GPUMemory::GPUString inStringCol, int8_t *inMask, int32_t inDataElementCount);

	/// Convert polygons to WKTs (GPUPolygon column to GPUString columns)
	/// <param name="outStringCol">output GPUString column</param>
	/// <param name="inPolygonCol">input GPUPolygon column</param>
	/// <param name="dataElementCount">input data element (complex polygon) count</param>
	static void ConvertPolyColToWKTCol(GPUMemory::GPUString *outStringCol,
		GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount);

	static void ConvertPointColToWKTCol(GPUMemory::GPUString * outStringCol, 
		NativeGeoPoint* inPointCol, int32_t dataElementCount);

	/// Recontruct GPUPolygon column and keep on GPU in the same format
	/// <param name="outCol">output GPUPolygon column</param>
	/// <param name="outDataElementCount">reconstructed data element (complex polygon) count</param>
	/// <param name="inCol">input GPUPolygon column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (complex polygon) count</param>
	static void ReconstructPolyColKeep(GPUMemory::GPUPolygon *outCol, int32_t *outDataElementCount,
		GPUMemory::GPUPolygon inCol, int8_t *inMask, int32_t inDataElementCount, int8_t** outNullMask = nullptr, int8_t* nullMask = nullptr);

	/// Reconstruct GPUPolygon column and convert to WKT string array on CPU
	/// <param name="outStringData">output CPU string array</param>
	/// <param name="outDataElementCount">reconstructed data element (WKT string) count</param>
	/// <param name="inPolygonCol">input GPUPolygon column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (complex polygon) count</param>
	static void ReconstructPolyColToWKT(std::string * outStringData, int32_t *outDataElementCount,
		GPUMemory::GPUPolygon inPolygonCol, int8_t *inMask, int32_t inDataElementCount, int8_t* outNullMask = nullptr, int8_t* nullMask = nullptr);

	/// Reconstruct NativeGeoPoint column and convert to WKT string array on CPU
	/// <param name="outStringData">output CPU string array</param>
	/// <param name="outDataElementCount">reconstructed data element (WKT string) count</param>
	/// <param name="inPointCol">input NativeGeoPoint column</param>
	/// <param name="inMask">input mask for the reconstruction</param>
	/// <param name="inDataElementCount">input data element (point) count</param>
	static void ReconstructPointColToWKT(std::string * outStringData, int32_t *outDataElementCount,
		NativeGeoPoint* inPointCol, int8_t *inMask, int32_t inDataElementCount, int8_t* outNullMask = nullptr, int8_t* nullMask = nullptr);

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
			if (*outDataElementCount > 0)
			{
				// Copy the generated output from GPU (device) to host
				GPUMemory::copyDeviceToHost(outData, outDataGPUPointer, *outDataElementCount);

				// Free the memory
				GPUMemory::free(outDataGPUPointer);
			}
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
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount, int8_t* outNullMask, int8_t* nullMask);

/// Specialization for ComplexPolygon (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructCol<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon *outData,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount, int8_t* outNullMask, int8_t* nullMask);


/// Specialization for Point (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::Point>(ColmnarDB::Types::Point **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount, int8_t** outNullMask, int8_t* nullMask);

/// Specialization for ComplexPolygon (not supported but need to be implemented)
template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount, int8_t** outNullMask, int8_t* nullMask);
