#include "GPUMemory.cuh"
#include "../../GpuSqlParser/GpuSqlDispatcher.h"
#include <vector>
#include <string>

bool GPUMemory::EvictWithLockList()
{
	return Context::getInstance().getCacheForCurrentDevice().evict();
}

void GPUMemory::clear()
{
	Context::getInstance().GetAllocatorForCurrentDevice().Clear();
	CheckCudaError(cudaGetLastError());
}

void GPUMemory::free(void *p_block)
{
	Context::getInstance().GetAllocatorForCurrentDevice().deallocate(static_cast<int8_t*>(p_block), 0);
	CheckCudaError(cudaGetLastError());
}