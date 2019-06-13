#include "GPUStringUnary.cuh"


__global__ void kernel_reverse_string(GPUMemory::GPUString outCol, GPUMemory::GPUString inCol, int64_t stringCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < stringCount; i += stride)
	{
		int64_t length = (i == 0) ? inCol.stringIndices[i] : (inCol.stringIndices[i] - inCol.stringIndices[i - 1]);
		int64_t index = (i == 0) ? 0 : inCol.stringIndices[i - 1];
		for (int32_t j = 0; j < length; j++)
		{
			outCol.allChars[index + j] = inCol.allChars[index + length - 1 - j];
		}
	}
}
