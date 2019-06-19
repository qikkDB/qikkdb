#include "GPUStringBinary.cuh"


__global__ void kernel_predict_length_concat(int32_t* newLengths, GPUMemory::GPUString inputA, bool isACol,
	GPUMemory::GPUString inputB, bool isBCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		newLengths[i] = GetStringLength(inputA.stringIndices, isACol ? i : 0) + GetStringLength(inputB.stringIndices, isBCol ? i : 0);
	}
}

__global__ void kernel_string_concat(GPUMemory::GPUString output, GPUMemory::GPUString inputA, bool isACol,
	GPUMemory::GPUString inputB, bool isBCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		const int32_t aI = isACol ? i : 0;
		const int64_t aIndex = GetStringIndex(inputA.stringIndices, aI);
		const int32_t aLength = static_cast<int32_t>(inputA.stringIndices[aI] - aIndex);
		const int32_t bI = isBCol ? i : 0;
		const int64_t bIndex = GetStringIndex(inputB.stringIndices, bI);
		const int32_t bLength = static_cast<int32_t>(inputB.stringIndices[bI] - bIndex);
		const int64_t outIndex = GetStringIndex(output.stringIndices, i);
		for (int32_t j = 0; j < aLength; j++)
		{
			output.allChars[outIndex + j] = inputA.allChars[aIndex + j];
		}
		for (int32_t j = 0; j < bLength; j++)
		{
			output.allChars[outIndex + aLength + j] = inputB.allChars[bIndex + j];
		}
	}
}
