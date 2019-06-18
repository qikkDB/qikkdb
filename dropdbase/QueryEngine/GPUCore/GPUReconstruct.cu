#include "GPUReconstruct.cuh"
#include "cuda_ptr.h"

// Polygon WKT format:
// POLYGON((179.9999 89.9999, 0.0000 0.0000, 179.9999 89.9999), (-179.9999 -89.9999, 52.1300 -27.0380, -179.9999 -89.9999))

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


/// Helping function to calculate number of digints of integer part of float
__device__ int32_t GetNumberOfIntegerPartDigits(float number)
{
	return (floorf(fabsf(number)) > 3.0f ?
		static_cast<int32_t>(log10f(floorf(fabsf(number)))) : 0) + 1 + (number < 0 ? 1 : 0);
}

__global__ void kernel_predict_wkt_lengths(int32_t * outStringLengths, GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// Count POLYGON word and parentheses ("POLYGON((), ())")
		int32_t charCounter = 11 + (4 * (inPolygonCol.polyCount[i] - 1));
		const int32_t subpolyStartIdx = inPolygonCol.polyIdx[i];
		const int32_t subpolyEndIdx = subpolyStartIdx + inPolygonCol.polyCount[i];
		for (int32_t j = subpolyStartIdx; j < subpolyEndIdx; j++)
		{
			const int32_t pointCount = inPolygonCol.pointCount[j] - 2;
			const int32_t pointStartIdx = inPolygonCol.pointIdx[j] + 1;
			const int32_t pointEndIdx = pointStartIdx + pointCount;

			// Count the decimal part and colons between points (".0000 .0000, .0000 .0000")
			charCounter += pointCount * (2 * WKT_DECIMAL_PLACES + 5) - 2;
			for (int32_t k = pointStartIdx; k < pointEndIdx; k++)
			{
				// Count the integer part ("150".0000, "-0".1000)
				charCounter += GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[k].latitude) +
					GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[k].longitude);
			}
		}
		outStringLengths[i] = charCounter;
	}
}


/// Helping function to "print" float to GPU char array
__device__ void FloatToString(char * allChars, int64_t &startIndex, float number)
{
	// Append sign
	if (number < 0)
	{
		allChars[startIndex] = '-';
		// (note that there is no addres move because we will count with negative sign later)
	}

	// Append integer part
	int32_t integerPart = static_cast<int32_t>(floorf(fabsf(number)));
	int32_t digits = GetNumberOfIntegerPartDigits(number);
	startIndex += digits;
	do
	{
		allChars[--startIndex] = ('0' + (integerPart % 10));
		integerPart /= 10;
	} while (integerPart > 0);		// Dynamic integer part places
	startIndex += digits - (number < 0 ? 1 : 0);

	// Append decimal part
	int32_t decimalPart = static_cast<int32_t>(roundf(fmodf(fabsf(number), 1.0f)*powf(10.0f, WKT_DECIMAL_PLACES)));
	allChars[startIndex++] = '.';
	startIndex += WKT_DECIMAL_PLACES;
	for (int32_t i = 0; i < WKT_DECIMAL_PLACES; i++)	// Fixed decimal places
	{
		allChars[--startIndex] = ('0' + (decimalPart % 10));
		decimalPart /= 10;
	}
	startIndex += WKT_DECIMAL_PLACES;
}


__global__ void kernel_convert_poly_to_wkt(GPUMemory::GPUString outWkt, GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)	// via complex polygons
	{
		// "POLYGON("
		const int64_t stringStartIndex = (i == 0 ? 0 : outWkt.stringIndices[i - 1]);
		for (int32_t j = 0; j < 7; j++)
		{
			outWkt.allChars[stringStartIndex + j] = WKT_POLYGON[j];
		}
		int64_t charId = stringStartIndex + 7;
		outWkt.allChars[charId++] = '(';

		const int32_t subpolyStartIdx = inPolygonCol.polyIdx[i];
		const int32_t subpolyEndIdx = subpolyStartIdx + inPolygonCol.polyCount[i];

		for (int32_t j = subpolyStartIdx; j < subpolyEndIdx; j++)	// via sub-polygons
		{
			outWkt.allChars[charId++] = '(';
			const int32_t pointCount = inPolygonCol.pointCount[j] - 2;
			const int32_t pointStartIdx = inPolygonCol.pointIdx[j] + 1;
			const int32_t pointEndIdx = pointStartIdx + pointCount;

			for (int32_t k = pointStartIdx; k < pointEndIdx; k++)	// via points
			{
				FloatToString(outWkt.allChars, charId, inPolygonCol.polyPoints[k].latitude);
				outWkt.allChars[charId++] = ' ';
				FloatToString(outWkt.allChars, charId, inPolygonCol.polyPoints[k].longitude);

				if (k < pointEndIdx - 1)
				{
					outWkt.allChars[charId++] = ',';
					outWkt.allChars[charId++] = ' ';
				}
			}

			outWkt.allChars[charId++] = ')';
			if (j < subpolyEndIdx - 1)
			{
				outWkt.allChars[charId++] = ',';
				outWkt.allChars[charId++] = ' ';
			}
		}
		outWkt.allChars[charId++] = ')';
		/*
		// Lengths mis-match check
		if (charId != outWkt.stringIndices[i])
		{
			printf("Not match fin id! %d\n", outWkt.stringIndices[i] - charId);
		}
		*/
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
		// todo later if needed
	}
	else	// If mask is not used
	{
		*outDataElementCount = inDataElementCount;
		if (inDataElementCount > 0)
		{
			// Copy string indices to host
			std::unique_ptr<int64_t[]> hostStringIndices = std::make_unique<int64_t[]>(inDataElementCount);
			GPUMemory::copyDeviceToHost(hostStringIndices.get(), inStringCol.stringIndices, inDataElementCount);
			int64_t fullCharCount = hostStringIndices[inDataElementCount - 1];

			// Copy all chars to host
			std::unique_ptr<char[]> hostAllChars = std::make_unique<char[]>(fullCharCount);
			GPUMemory::copyDeviceToHost(hostAllChars.get(), inStringCol.allChars, fullCharCount);

			// Fill output string array
			for (int32_t i = 0; i < inDataElementCount; i++)
			{
				size_t length = static_cast<size_t>(i == 0 ? hostStringIndices[0] :
					hostStringIndices[i] - hostStringIndices[i - 1]);
				outStringData[i] = std::string(hostAllChars.get() +
					(i == 0 ? 0 : hostStringIndices[i - 1]), length);
			}
		}
	}
}


void GPUReconstruct::ConvertPolyColToWKTCol(GPUMemory::GPUString *outStringCol,
	GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
	Context& context = Context::getInstance();
	if (dataElementCount > 0)
	{
		// "Predict" (pre-calculate) string lengths
		cuda_ptr<int32_t>stringLengths(dataElementCount);
		kernel_predict_wkt_lengths << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(stringLengths.get(), inPolygonCol, dataElementCount);
		CheckCudaError(cudaGetLastError());

		// Alloc and compute string indices as a prefix sum of the string lengths
		GPUMemory::alloc(&(outStringCol->stringIndices), dataElementCount);
		PrefixSum(outStringCol->stringIndices, stringLengths.get(), dataElementCount);

		// Get total char count and alloc array for all chars
		int64_t totalCharCount;
		GPUMemory::copyDeviceToHost(&totalCharCount, outStringCol->stringIndices + dataElementCount - 1, 1);
		GPUMemory::alloc(&(outStringCol->allChars), totalCharCount);

		// Finally convert polygons to WKTs
		kernel_convert_poly_to_wkt << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(*outStringCol, inPolygonCol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}
	else
	{
		outStringCol->allChars = nullptr;
		outStringCol->stringIndices = nullptr;
	}
}


void GPUReconstruct::ReconstructPolyColKeep(GPUMemory::GPUPolygon *outCol, int32_t *outDataElementCount,
	GPUMemory::GPUPolygon inCol, int8_t *inMask, int32_t inDataElementCount)
{
	Context& context = Context::getInstance();

	if (inMask)		// If mask is used (if inMask is not nullptr)
	{
		// A buffer for the prefix sum vector
		cuda_ptr<int32_t> inPrefixSumPointer(inDataElementCount);
		PrefixSum(inPrefixSumPointer.get(), inMask, inDataElementCount);
		GPUMemory::copyDeviceToHost(outDataElementCount, inPrefixSumPointer.get() + inDataElementCount - 1, 1);

		if (*outDataElementCount > 0)	// Not empty result set
		{
			// Reconstruct each array independently
			int32_t inSubpolySize = CalculateCount(inCol.polyIdx, inCol.polyCount, inDataElementCount);
			int32_t inPointSize = CalculateCount(inCol.pointIdx, inCol.pointCount, inSubpolySize);

			// Complex polygons (reconstruct polyCount and sum it to polyIdx)
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

			cuda_ptr<int32_t> pointPrefixSumPointer(inPointSize);
			PrefixSum(pointPrefixSumPointer.get(), pointMask.get(), inPointSize);

			GPUMemory::alloc(&(outCol->polyPoints), outPointSize);
			kernel_reconstruct_col << < context.calcGridDim(inSubpolySize), context.getBlockDim() >> >
				(outCol->polyPoints, inCol.polyPoints, pointPrefixSumPointer.get(), pointMask.get(), inPointSize);
			CheckCudaError(cudaGetLastError());
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
	GPUMemory::GPUString gpuWkt;
	ConvertPolyColToWKTCol(&gpuWkt, reconstructedPolygonCol, *outDataElementCount);
	GPUMemory::free(reconstructedPolygonCol);
	// Use reconstruct without mask - just to convert GPUString to CPU string array
	ReconstructStringCol(outStringData, outDataElementCount, gpuWkt, nullptr, *outDataElementCount);
	GPUMemory::free(gpuWkt);
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
