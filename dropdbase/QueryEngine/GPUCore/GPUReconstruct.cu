#include "GPUReconstruct.cuh"
#include "cuda_ptr.h"

__global__ void kernel_generate_subpoly_mask(int8_t *outMask, int8_t *inMask, int32_t *polyIdx, int32_t *polyCount, int32_t polyIdxSize)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < polyIdxSize; i += stride)
	{
		for (int32_t j = 0; j < polyCount[i]; j++)
		{
			outMask[polyIdx[i] + j] = inMask[i];
		}
	}
}

void GPUReconstruct::ReconstructPolyCol(GPUMemory::GPUPolygon outData, int32_t *outDataElementCount,
	GPUMemory::GPUPolygon ACol, int8_t *inMask, int32_t dataElementCount)
{

}

void GPUReconstruct::ReconstructPolyColKeep(GPUMemory::GPUPolygon *outCol, int32_t *outDataElementCount,
	GPUMemory::GPUPolygon ACol, int8_t *inMask, int32_t dataElementCount)
{
	Context& context = Context::getInstance();

	if (inMask)		// If inMask is not nullptr
	{
		int32_t* prefixSumPointer = nullptr;
		try
		{
			// Malloc a new buffer for the prefix sum vector
			GPUMemory::alloc(&prefixSumPointer, dataElementCount);

			PrefixSum(prefixSumPointer, inMask, dataElementCount);
			GPUMemory::copyDeviceToHost(outDataElementCount, prefixSumPointer + dataElementCount - 1, 1);
			if (*outDataElementCount > 0)
			{
				GPUMemory::alloc(&(outCol->polyCount), *outDataElementCount);
				GPUMemory::alloc(&(outCol->polyIdx), *outDataElementCount);
				// Reconstruct each array independently
				kernel_reconstruct_col << < context.calcGridDim(*outDataElementCount), context.getBlockDim() >> >
					(outCol->polyCount, ACol.polyCount, prefixSumPointer, inMask, dataElementCount);
				PrefixSumExclusive(outCol->polyIdx, outCol->polyCount, dataElementCount);
				
				int32_t totalSubpolySize;
				int32_t lastSubpolySize;
				GPUMemory::copyDeviceToHost(&totalSubpolySize, outCol->polyIdx + dataElementCount - 1, 1);
				GPUMemory::copyDeviceToHost(&lastSubpolySize, outCol->polyCount + dataElementCount - 1, 1);
				totalSubpolySize += lastSubpolySize;

				cuda_ptr<int8_t> subpolyMask(totalSubpolySize);
				kernel_generate_subpoly_mask << < context.calcGridDim(*outDataElementCount), context.getBlockDim() >> >
					(subpolyMask.get(), inMask, outCol->polyIdx, outCol->polyCount, *outDataElementCount);
				// TODO subpoly (pointCount and pointIdx) reconstruction based on subpolyMask
				// and then polyPoints reconstruction
			}
			else
			{
				*outCol = ACol;
			}
			// Free the memory
			GPUMemory::free(prefixSumPointer);
		}
		catch (...)
		{
			if (prefixSumPointer)
			{
				GPUMemory::free(prefixSumPointer);
			}

			throw;
		}
	}
	else	// If inMask is nullptr, just copy pointers from ACol to outCol
	{
		*outCol = ACol;
		*outDataElementCount = dataElementCount;
	}

	// Get last error
	CheckCudaError(cudaGetLastError());
}


template<>
void GPUReconstruct::reconstructCol<ColmnarDB::Types::Point>(ColmnarDB::Types::Point *outData,
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount)
{
	// Not supported, just throw an error
	CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
		"ReconstructCol of Point not supported, use GenerateIndexes instead");
}

template<>
void GPUReconstruct::reconstructCol<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon *outData,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount)
{
	// Not supported, just throw an error
	CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
		"ReconstructCol of ComplexPolygon not supported, use GenerateIndexes instead");
}

template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::Point>(ColmnarDB::Types::Point **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::Point *ACol, int8_t *inMask, int32_t dataElementCount)
{
	// Not supported, just throw an error
	CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
		"ReconstructColKeep of Point not supported, use GenerateIndexes instead");
}

template<>
void GPUReconstruct::reconstructColKeep<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon **outCol,
	int32_t *outDataElementCount, ColmnarDB::Types::ComplexPolygon *ACol, int8_t *inMask, int32_t dataElementCount)
{
	// Not supported, just throw an error
	CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
		"ReconstructColKeep of ComplexPolygon not supported, use GenerateIndexes instead");
}
