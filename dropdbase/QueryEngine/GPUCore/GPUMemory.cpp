#include "GPUMemory.cuh"
#include "../../GpuSqlParser/GpuSqlDispatcher.h"
#include <vector>
#include <string>

bool GPUMemory::EvictWithLockList()
{
	return Context::getInstance().getCacheForCurrentDevice().evict();
}