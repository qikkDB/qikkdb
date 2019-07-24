#include "GPUOrderBy.cuh"

// Fill the index buffers with default indices
__global__ void kernel_fill_indices(int32_t* indices, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		indices[i] = i;
	}
}

GPUOrderBy::GPUOrderBy(int32_t dataElementCount)
{
	GPUMemory::alloc(&indices1, dataElementCount);
	GPUMemory::alloc(&indices2, dataElementCount);

	// Initialize the index buffer
	kernel_fill_indices << < Context::getInstance().calcGridDim(dataElementCount),
		Context::getInstance().getBlockDim() >> >
		(indices1, dataElementCount);
}

GPUOrderBy::~GPUOrderBy()
{
	GPUMemory::free(indices1);
	GPUMemory::free(indices2);
}
