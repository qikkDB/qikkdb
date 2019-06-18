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

void GPUMemory::free(GPUPolygon polygonCol)
{
	GPUMemory::free(polygonCol.polyPoints);
	GPUMemory::free(polygonCol.pointIdx);
	GPUMemory::free(polygonCol.pointCount);
	GPUMemory::free(polygonCol.polyIdx);
	GPUMemory::free(polygonCol.polyCount);
}

void GPUMemory::free(GPUString stringCol)
{
	GPUMemory::free(stringCol.allChars);
	GPUMemory::free(stringCol.stringIndices);
}
