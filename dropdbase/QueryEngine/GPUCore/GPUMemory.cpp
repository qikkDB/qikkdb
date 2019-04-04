#include "GpuMemory.cuh"
#include "../../GpuSqlParser/GpuSqlDispatcher.h"
#include <vector>
#include <string>

bool GPUMemory::EvictWithLockList()
{
	std::vector<std::string> lockList;
	for (auto& table : GpuSqlDispatcher::linkTable)
	{
		lockList.push_back(table.first);
	}
	return Context::getInstance().getCacheForCurrentDevice().evict(lockList);
}