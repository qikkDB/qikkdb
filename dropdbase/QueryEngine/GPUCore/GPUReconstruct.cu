#include "GPUReconstruct.cuh"
#include "cuda_ptr.h"

__global__ void kernel_generate_submask(int8_t *outMask, int8_t *inMask, int32_t *indices, int32_t *counts, int32_t size)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < size; i += stride)
	{
		for (int32_t j = 0; j < counts[i]; j++)
		{
			outMask[indices[i] + j] = inMask[i];
		}
	}
}

int32_t GPUReconstruct::CalculateCount(int32_t * indices, int32_t * counts, int32_t size)
{
	int32_t lastIndex;
	int32_t lastCount;
	GPUMemory::copyDeviceToHost(&lastIndex, indices + size - 1, 1);
	GPUMemory::copyDeviceToHost(&lastCount, counts + size - 1, 1);
	return lastIndex + lastCount;
}

void GPUReconstruct::ReconstructStringCol(std::string *outStringData, int32_t *outDataElementCount,
	GPUMemory::GPUString inStringCol, int8_t *inMask, int32_t inDataElementCount)
{

	if (inMask)		// If mask is used (if inMask is not nullptr)
	{
		CheckQueryEngineError(GPU_EXTENSION_ERROR, "ReconstructStringCol with mask not implemented yet");
	}
	else	// If mask is not used
	{
		*outDataElementCount = inDataElementCount;
		std::unique_ptr<int32_t[]> hostStringStarts = std::make_unique<int32_t[]>(inDataElementCount);
		std::unique_ptr<int32_t[]> hostStringLengths = std::make_unique<int32_t[]>(inDataElementCount);
		GPUMemory::copyDeviceToHost(hostStringStarts.get(), inStringCol.stringStarts, inDataElementCount);
		GPUMemory::copyDeviceToHost(hostStringLengths.get(), inStringCol.stringLengths, inDataElementCount);
		int32_t fullCharCount = hostStringStarts[inDataElementCount - 1] +
								hostStringLengths[inDataElementCount - 1];

		std::unique_ptr<char[]> hostAllChars = std::make_unique<char[]>(fullCharCount);
		GPUMemory::copyDeviceToHost(hostAllChars.get(), inStringCol.allChars, fullCharCount);

		for (int32_t i = 0; i < inDataElementCount; i++)
		{
			outStringData[i] = std::string(hostAllChars.get() + hostStringStarts[i],
				static_cast<size_t>(hostStringLengths[i]));
		}
	}
}

void GPUReconstruct::ConvertPolyColToWKTCol(GPUMemory::GPUString *outStringCol,
	GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
	// TODO
}

void GPUReconstruct::ReconstructPolyColKeep(GPUMemory::GPUPolygon *outCol, int32_t *outDataElementCount,
	GPUMemory::GPUPolygon inCol, int8_t *inMask, int32_t inDataElementCount)
{
	Context& context = Context::getInstance();

	if (inMask)		// If mask is used (if inMask is not nullptr)
	{
		// Malloc a new buffer for the prefix sum vector
		cuda_ptr<int32_t> inPrefixSumPointer(inDataElementCount);
		PrefixSum(inPrefixSumPointer.get(), inMask, inDataElementCount);
		GPUMemory::copyDeviceToHost(outDataElementCount, inPrefixSumPointer.get() + inDataElementCount - 1, 1);

		if (*outDataElementCount > 0)	// Not empty result set
		{
			// Reconstruct each array independently
			int32_t inSubpolySize = CalculateCount(inCol.polyIdx, inCol.polyCount, inDataElementCount);
			int32_t inPointSize = CalculateCount(inCol.pointIdx, inCol.pointCount, inSubpolySize);

			// Comlex polygons (reconstruct polyCount and sum it to polyIdx)
			GPUMemory::alloc(&(outCol->polyCount), *outDataElementCount);
			GPUMemory::alloc(&(outCol->polyIdx), *outDataElementCount);
			kernel_reconstruct_col << < context.calcGridDim(inDataElementCount), context.getBlockDim() >> >
				(outCol->polyCount, inCol.polyCount, inPrefixSumPointer.get(), inMask, inDataElementCount);
			CheckCudaError(cudaGetLastError());
			PrefixSumExclusive(outCol->polyIdx, outCol->polyCount, *outDataElementCount);

			// Subpolygons (reconstruct pointCount and sum it to pointIdx)
			int32_t outSubpolySize = CalculateCount(outCol->polyIdx, outCol->polyCount, *outDataElementCount);

			cuda_ptr<int8_t> subpolyMask(inSubpolySize);
			kernel_generate_submask << < context.calcGridDim(inDataElementCount), context.getBlockDim() >> >
				(subpolyMask.get(), inMask, inCol.polyIdx, inCol.polyCount, inDataElementCount);
			CheckCudaError(cudaGetLastError());
			int8_t spm[1000];
			GPUMemory::copyDeviceToHost(spm, subpolyMask.get(), inSubpolySize);

			cuda_ptr<int32_t> subpolyPrefixSumPointer(inSubpolySize);
			PrefixSum(subpolyPrefixSumPointer.get(), subpolyMask.get(), inSubpolySize);

			GPUMemory::alloc(&(outCol->pointCount), outSubpolySize);
			GPUMemory::alloc(&(outCol->pointIdx), outSubpolySize);
			kernel_reconstruct_col << < context.calcGridDim(inSubpolySize), context.getBlockDim() >> >
				(outCol->pointCount, inCol.pointCount, subpolyPrefixSumPointer.get(), subpolyMask.get(), inSubpolySize);
			CheckCudaError(cudaGetLastError());
			PrefixSumExclusive(outCol->pointIdx, outCol->pointCount, outSubpolySize);

			// Points (reconstruct polyPoints)
			int32_t outPointSize = CalculateCount(outCol->pointIdx, outCol->pointCount, outSubpolySize);

			cuda_ptr<int8_t> pointMask(inPointSize);
			kernel_generate_submask << < context.calcGridDim(inSubpolySize), context.getBlockDim() >> >
				(pointMask.get(), subpolyMask.get(), inCol.pointIdx, inCol.pointCount, inSubpolySize);
			CheckCudaError(cudaGetLastError());
			int8_t pm[1000];
			GPUMemory::copyDeviceToHost(pm, pointMask.get(), inPointSize);

			cuda_ptr<int32_t> pointPrefixSumPointer(inPointSize);
			PrefixSum(pointPrefixSumPointer.get(), pointMask.get(), inPointSize);

			GPUMemory::alloc(&(outCol->polyPoints), outPointSize);
			kernel_reconstruct_col << < context.calcGridDim(inSubpolySize), context.getBlockDim() >> >
				(outCol->polyPoints, inCol.polyPoints, pointPrefixSumPointer.get(), pointMask.get(), inPointSize);
			CheckCudaError(cudaGetLastError());
			NativeGeoPoint ngp[1000];
			GPUMemory::copyDeviceToHost(ngp, outCol->polyPoints, outPointSize);
		}
		else	// Empty result set
		{
			outCol->polyPoints = nullptr;
			outCol->pointIdx = nullptr;
			outCol->pointCount = nullptr;
			outCol->polyIdx = nullptr;
			outCol->polyCount = nullptr;
		}
	}
	else	// If mask is not used (is nullptr), just copy pointers from inCol to outCol
	{
		*outCol = inCol;
		*outDataElementCount = inDataElementCount;
	}

	// Get last error
	CheckCudaError(cudaGetLastError());
}

void GPUReconstruct::ReconstructPolyColToWKT(std::string *outStringData, int32_t *outDataElementCount,
	GPUMemory::GPUPolygon inPolygonCol, int8_t *inMask, int32_t inDataElementCount)
{
	GPUMemory::GPUPolygon reconstructedPolygonCol;
	ReconstructPolyColKeep(&reconstructedPolygonCol, outDataElementCount, inPolygonCol, inMask, inDataElementCount);
	// TODO convert to WKT and "reconstruct" string
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
