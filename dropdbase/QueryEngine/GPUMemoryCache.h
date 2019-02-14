#pragma once

#include <unordered_map>
#include <set>
#include <string>
#include <list>
#include <stdexcept>
#include "CudaMemAllocator.h"

class GPUMemoryCache
{

private:
	const size_t maxSize = 7000000000LL;
	//const size_t maxSize = 9200;
	int32_t deviceID_;

	struct CacheEntry
	{
		std::uintptr_t ptr;
		size_t size;
		std::list<std::unordered_map<std::string, CacheEntry>::iterator>::iterator lruQueueIt;
	};

	std::unordered_map<std::string, CacheEntry> cacheMap;
	std::list<std::unordered_map<std::string, CacheEntry>::iterator> lruQueue;

	int64_t usedSize;
	void evict();

	CudaMemAllocator& GetAllocator();

	bool tryInsert(size_t sizeToInsert) const
	{
		return usedSize + sizeToInsert < maxSize;
	}

public:

	GPUMemoryCache(int32_t deviceID);
	~GPUMemoryCache();
	template<typename T>
	std::tuple<T*, size_t, bool> getColumn(const std::string& columnName, int32_t blockIndex, size_t size)
	{
		std::string columnBlock = columnName + "_" + std::to_string(blockIndex);
		if (cacheMap.find(columnBlock) != cacheMap.end())
		{
			lruQueue.erase(cacheMap.at(columnBlock).lruQueueIt);
			lruQueue.push_back(cacheMap.find(columnBlock));
			cacheMap.at(columnBlock).lruQueueIt = (--lruQueue.end());
			return { reinterpret_cast<T*>(cacheMap.at(columnBlock).ptr), cacheMap.at(columnBlock).size / sizeof(T), true };
		}
		size_t sizeToInsert = sizeof(T) * size;

		if (sizeToInsert > maxSize)
		{
			throw std::length_error("Tried to cache block larger than maximum cache size");
		}

		while (!tryInsert(sizeToInsert)) 
		{
			evict();
		}

		T* newPtr = reinterpret_cast<T*>(GetAllocator().allocate(size*sizeof(T)));
		usedSize += sizeToInsert;
		CacheEntry newCacheEntry{ reinterpret_cast<std::uintptr_t>(newPtr), sizeToInsert, lruQueue.end() };
		auto cacheMapIt = cacheMap.insert(std::make_pair(columnBlock, std::move(newCacheEntry))).first;
		lruQueue.push_back(cacheMapIt);
		cacheMapIt->second.lruQueueIt = (--lruQueue.end());
		return { newPtr, size, false };
	}

	void clearCachedBlock(const std::string& columnName, int32_t blockIndex);
	bool containsColumn(const std::string& columnName, int32_t blockIndex);
	GPUMemoryCache(const GPUMemoryCache&) = delete;
	GPUMemoryCache& operator=(const GPUMemoryCache&) = delete;
};
