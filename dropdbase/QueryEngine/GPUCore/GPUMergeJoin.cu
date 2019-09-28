#include "GPUMergeJoin.cuh"

__global__ void kernel_label_input(int32_t *colBlockIndices, int32_t blockOffset, int32_t dataElementCount) 
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		colBlockIndices[i] = blockOffset + i;
	}
}