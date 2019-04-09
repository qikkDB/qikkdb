#pragma once

#include <unordered_map>
#include <set>
#include <string>
#include <list>
#include <stdexcept>
#include <vector>
#include "CudaMemAllocator.h"

class GPUMemoryCache
{

private:
	const size_t maxSize_;
	int32_t deviceID_;
	struct CacheEntry;

	struct CacheEntryRefWrapper
	{
		CacheEntryRefWrapper(CacheEntry& entry) :
			ref(entry)
		{
		}
		CacheEntry& ref;
	};

	struct CacheEntry
	{
		std::string key;
		std::uintptr_t ptr;
		size_t size;
		std::list<CacheEntryRefWrapper>::iterator lruQueueIt;
	};



	std::unordered_map<std::string, CacheEntry> cacheMap;
	std::list<CacheEntryRefWrapper> lruQueue;

	int64_t usedSize;

	CudaMemAllocator& GetAllocator();

	bool tryInsert(size_t sizeToInsert) const
	{
		return usedSize + sizeToInsert <= maxSize_;
	}
	///	<summary>
	/// List of columns that must not be evicted
	/// </summary>
	static std::vector<std::string> lockList;
public:
	static void SetLockList(const std::vector<std::string>& lockList);
	GPUMemoryCache(int32_t deviceID, size_t maximumSize);
	~GPUMemoryCache();
	template<typename T>
	std::tuple<T*, size_t, bool> getColumn(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex, size_t size)
	{
		std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex);
		if (cacheMap.find(columnBlock) != cacheMap.end())
		{
			lruQueue.erase(cacheMap.at(columnBlock).lruQueueIt);
			lruQueue.push_back(cacheMap.at(columnBlock));
			cacheMap.at(columnBlock).lruQueueIt = (--lruQueue.end());
			return { reinterpret_cast<T*>(cacheMap.at(columnBlock).ptr), cacheMap.at(columnBlock).size / sizeof(T), true };
		}
		size_t sizeToInsert = sizeof(T) * size;

		if (sizeToInsert > maxSize_)
		{
			throw std::length_error("Tried to cache block larger than maximum cache size");
		}

		while (!tryInsert(sizeToInsert)) 
		{
			evict();
		}

		T* newPtr = reinterpret_cast<T*>(GetAllocator().allocate(size*sizeof(T)));
		usedSize += sizeToInsert;
		CacheEntry newCacheEntry{ columnBlock, reinterpret_cast<std::uintptr_t>(newPtr), sizeToInsert, lruQueue.end() };
		auto cacheMapIt = cacheMap.insert(std::make_pair(columnBlock, std::move(newCacheEntry))).first;
		lruQueue.emplace_back(cacheMapIt->second);
		cacheMapIt->second.lruQueueIt = (--lruQueue.end());
		return { newPtr, size, false };
	}
	bool evict();
	void clearCachedBlock(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex);
	bool containsColumn(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex);
	GPUMemoryCache(const GPUMemoryCache&) = delete;
	GPUMemoryCache& operator=(const GPUMemoryCache&) = delete;
};
