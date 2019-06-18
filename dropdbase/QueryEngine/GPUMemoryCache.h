#pragma once

#include <unordered_map>
#include <string>
#include <list>
#include <stdexcept>
#include <vector>
#include "CudaMemAllocator.h"

class GPUMemoryCache
{

private:
	/// <summary>
	/// Maximum size of cache in bytes
	/// </summary>
	const size_t maxSize_;

	/// <summary>
	/// Device to which current cache belongs
	/// </summary>
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

	/// <summary>
	/// Allocations by name
	/// </summary>
	std::unordered_map<std::string, CacheEntry> cacheMap;
	/// <summary>
	/// Allocations ordered by eviction priority
	/// </summary>
	std::list<CacheEntryRefWrapper> lruQueue;

	/// <summary>
	/// Number of bytes currently used by cache
	/// </summary>
	int64_t usedSize;
	/// <summary>
	/// Get allocator for the device to which this cache belongs
	/// </summary>
	/// <returns>Allocator object</returns>
	CudaMemAllocator& GetAllocator();

	/// <summary>
	/// Check if given number of bytes can be inserted into cache
	/// </summary>
	/// <param name="sizeToInsert">Number of bytes to check</param>
	/// <returns>true if there is enough free space in cache, otherwise false</returns>
	bool tryInsert(size_t sizeToInsert) const
	{
		return usedSize + sizeToInsert <= maxSize_;
	}

	/// <summary>List of columns that must not be evicted</summary>
	static std::vector<std::string> lockList;

public:
	/// <summary>
	/// Set list of column names that cannot be evicted
	/// </summary>
	/// <param name="lockList">List of column names</param>
	static void SetLockList(const std::vector<std::string>& lockList);

	/// <summary>
	/// Create cache of given size on given device
	/// </summary>
	/// <param name="deviceID">Device for which to create cache</param>
	/// <param name="maximumSize">Maximum size of cache in bytes</param>
	GPUMemoryCache(int32_t deviceID, size_t maximumSize);
	~GPUMemoryCache();

	/// <summary>
	/// Get column from cache or allocate new one
	/// </summary>
	/// <param name="databaseName">Database of cached column</param>
	/// <param name="tableAndColumnName">table and column name of cached column</param>
	/// <param name="blockIndex">Block index of cached column</param>
	/// <param name="size">Number of elements in block</param>
	/// <returns>Tuple containing pointer, size of cached block, and whether it is fresh allocation or cache hit</returns>
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
			if(!evict())
			{
				throw std::length_error("Not enough space left in cache");
			}
		}

		T* newPtr = reinterpret_cast<T*>(GetAllocator().allocate(size*sizeof(T)));
		usedSize += sizeToInsert;
		CacheEntry newCacheEntry{ columnBlock, reinterpret_cast<std::uintptr_t>(newPtr), sizeToInsert, lruQueue.end() };
		auto cacheMapIt = cacheMap.insert(std::make_pair(columnBlock, std::move(newCacheEntry))).first;
		lruQueue.emplace_back(cacheMapIt->second);
		cacheMapIt->second.lruQueueIt = (--lruQueue.end());
		return { newPtr, size, false };
	}
	/// <summary>
	/// Try to evict least recently used block, taking lock list into consideration
	/// </summary>
	/// <returns>True if something was evicted otherwise false</returns>
	bool evict();
	/// <summary>
	/// Remove specific block from cache
	/// </summary>
	/// <param name="databaseName">Database of cached column</param>
	/// <param name="tableAndColumnName">table and column name of cached column</param>
	/// <param name="blockIndex">Block index of cached column</param>
	void clearCachedBlock(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex);
	
	/// <summary>
	/// Check if block is in cache
	/// </summary>
	/// <param name="databaseName">Database of cached column</param>
	/// <param name="tableAndColumnName">table and column name of cached column</param>
	/// <param name="blockIndex">Block index of cached column</param>
	/// <returns>True if block is cached, otherwise false</returns>
	bool containsColumn(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex);

	GPUMemoryCache(const GPUMemoryCache&) = delete;
	GPUMemoryCache& operator=(const GPUMemoryCache&) = delete;
};
