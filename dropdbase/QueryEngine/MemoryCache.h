#pragma once

#include <unordered_map>
#include <set>
#include <string>
#include "GPUCore/GPUMemory.cuh"

class MemoryCache
{

private:
	const int64_t maxSize = 1 << 20;

	std::unordered_map<std::string, std::pair<uintptr_t, int32_t>> cacheMap;

	int64_t usedSize;
	void evict();

	bool tryInsert(int32_t sizeToInsert)
	{
		return usedSize + sizeToInsert < maxSize;
	}

public:

	MemoryCache();

	template<typename T>
	std::pair<uintptr_t, int32_t> getColumn(const std::string& columnName, int32_t blockIndex, int32_t size)
	{
		std::string columnBlock = columnName + "_" + std::to_string(blockIndex);
		if (cacheMap.find(columnBlock) != cacheMap.end())
		{
			return cacheMap.at(columnBlock);
		}

		int32_t sizeToInsert = sizeof(T) * size;

		while (!tryInsert(sizeToInsert)) 
		{
			evict();
		}

		T* newPtr;
		GPUMemory::alloc(&newPtr, size);
		std::pair newPair = std::make_pair(reinterpret_cast<uintptr_t>(newPtr), size);
		usedSize += sizeToInsert;

		cacheMap.insert({ columnBlock, newPair });
		return newPair;
	}
};
