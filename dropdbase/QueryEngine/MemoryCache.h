#pragma once

#include <unordered_map>
#include <string>
#include "GPUCore/GPUMemory.cuh"

class MemoryCache
{

private:
	std::unordered_map<std::string, std::pair<std::pair<uintptr_t, int32_t>, int32_t>> cacheMap;

public:

	template<typename T>
	std::pair<uintptr_t, int32_t> getColumn(const std::string& columnName, int32_t blockIndex, int32_t size)
	{
		std::string columnBlock = columnName + "_" + std::to_string(blockIndex);
		if (cacheMap.find(columnBlock) != cacheMap.end())
		{
			cacheMap.at(columnBlock).second++;
			return cacheMap.at(columnBlock).first;
		}

		T* newPtr;
		GPUMemory::alloc(&newPtr, size);
		cacheMap.insert({ columnBlock, std::make_pair(std::make_pair(reinterpret_cast<uintptr_t>(newPtr), size), 0) });
	}
};
