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
		// Malloc a new buffer for the prefix sum vector
		cuda_ptr<int32_t> polyPrefixSumPointer(dataElementCount);
		PrefixSum(polyPrefixSumPointer.get(), inMask, dataElementCount);
		GPUMemory::copyDeviceToHost(outDataElementCount, polyPrefixSumPointer.get() + dataElementCount - 1, 1);
		if (*outDataElementCount > 0)
		{
			int32_t totalInSubpolySize;
			int32_t lastInSubpolySize;
			GPUMemory::copyDeviceToHost(&totalInSubpolySize, ACol.polyIdx + dataElementCount - 1, 1);
			GPUMemory::copyDeviceToHost(&lastInSubpolySize, ACol.polyCount + dataElementCount - 1, 1);
			totalInSubpolySize += lastInSubpolySize;

			// Reconstruct each array independently
			GPUMemory::alloc(&(outCol->polyCount), *outDataElementCount);
			GPUMemory::alloc(&(outCol->polyIdx), *outDataElementCount);

			// Reconstruct polyCount and sum it to polyIdx
			kernel_reconstruct_col << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(outCol->polyCount, ACol.polyCount, polyPrefixSumPointer.get(), inMask, dataElementCount);
			PrefixSumExclusive(outCol->polyIdx, outCol->polyCount, *outDataElementCount);

			int32_t totalOutSubpolySize;
			int32_t lastOutSubpolySize;
			GPUMemory::copyDeviceToHost(&totalOutSubpolySize, outCol->polyIdx + *outDataElementCount - 1, 1);
			GPUMemory::copyDeviceToHost(&lastOutSubpolySize, outCol->polyCount + *outDataElementCount - 1, 1);
			totalOutSubpolySize += lastOutSubpolySize;

			cuda_ptr<int8_t> subpolyMask(totalInSubpolySize);
			kernel_generate_subpoly_mask << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
				(subpolyMask.get(), inMask, ACol.polyIdx, ACol.polyCount, dataElementCount);

			GPUMemory::alloc(&(outCol->pointCount), totalOutSubpolySize);
			GPUMemory::alloc(&(outCol->pointIdx), totalOutSubpolySize);
			cuda_ptr<int32_t> pointPrefixSumPointer(totalInSubpolySize);
			PrefixSum(pointPrefixSumPointer.get(), subpolyMask.get(), totalInSubpolySize);

			kernel_reconstruct_col << < context.calcGridDim(totalInSubpolySize), context.getBlockDim() >> >
				(outCol->pointCount, ACol.pointCount, pointPrefixSumPointer.get(), subpolyMask.get(), totalInSubpolySize);
			PrefixSumExclusive(outCol->pointIdx, outCol->pointCount, totalOutSubpolySize);

			int32_t totalOutPointSize;
			int32_t lastOutPointSize;
			GPUMemory::copyDeviceToHost(&totalOutPointSize, outCol->pointIdx + totalOutSubpolySize - 1, 1);
			GPUMemory::copyDeviceToHost(&lastOutPointSize, outCol->pointCount + totalOutSubpolySize - 1, 1);
			totalOutPointSize += lastOutPointSize;

			int32_t totalInPointSize;
			int32_t lastInPointSize;
			GPUMemory::copyDeviceToHost(&totalInPointSize, ACol.pointIdx + totalInSubpolySize - 1, 1);  //TODO
			GPUMemory::copyDeviceToHost(&lastInPointSize, ACol.pointCount + totalInSubpolySize - 1, 1); //TODO
			totalInPointSize += lastInPointSize;

			GPUMemory::alloc(&(outCol->polyPoints), totalOutPointSize);
			cuda_ptr<int32_t> polyPointsPrefixSumPointer(totalInSubpolySize);
			PrefixSum(polyPointsPrefixSumPointer.get(), subpolyMask.get(), totalInSubpolySize);

			kernel_reconstruct_col << < context.calcGridDim(totalInSubpolySize), context.getBlockDim() >> >
				(outCol->polyPoints, ACol.polyPoints, pointPrefixSumPointer.get(), subpolyMask.get(), totalInSubpolySize); // TODO
		}
		else
		{
			*outCol = ACol;
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
