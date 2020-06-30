#pragma once

#include <map>
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
        CacheEntryRefWrapper(CacheEntry& entry) : ref(entry)
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
    std::map<std::string, CacheEntry> cacheMap_;
    /// <summary>
    /// Allocations ordered by eviction priority
    /// </summary>
    std::list<CacheEntryRefWrapper> lruQueue_;

    std::map<std::string, CacheEntry>::iterator FindPrefix(const std::string& search_for);


    /// <summary>
    /// Number of bytes currently used by cache
    /// </summary>
    int64_t usedSize_;
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
    bool HasFreeSpace(size_t sizeToInsert) const
    {
        return usedSize_ + sizeToInsert <= maxSize_;
    }

    /// <summary>List of columns that must not be evicted</summary>
    static std::vector<std::string> lockList_;
    size_t currentBlockIndex_;

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
    template <typename T>
    std::tuple<T*, size_t, bool> GetColumn(const std::string& databaseName,
                                           const std::string& tableAndColumnName,
                                           int32_t blockIndex,
                                           size_t size,
                                           int64_t loadSize,
                                           int64_t loadOffset)
    {
        std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex) +
                                  "_" + std::to_string(loadSize) + "_" + std::to_string(loadOffset);
        if (cacheMap_.find(columnBlock) != cacheMap_.end())
        {
            lruQueue_.erase(cacheMap_.at(columnBlock).lruQueueIt);
            lruQueue_.push_back(cacheMap_.at(columnBlock));
            cacheMap_.at(columnBlock).lruQueueIt = (--lruQueue_.end());
            return {reinterpret_cast<T*>(cacheMap_.at(columnBlock).ptr),
                    cacheMap_.at(columnBlock).size / sizeof(T), true};
        }
        const size_t sizeToInsert = sizeof(T) * size;

        if (sizeToInsert > maxSize_)
        {
            throw std::length_error("Tried to cache block larger than maximum cache size");
        }

        while (!HasFreeSpace(sizeToInsert) || !GetAllocator().CanAllocate(sizeToInsert))
        {
            if (!Evict())
            {
                throw std::length_error("Not enough space left in cache");
            }
        }

        T* newPtr = reinterpret_cast<T*>(GetAllocator().Allocate(sizeToInsert));
        usedSize_ += sizeToInsert;
        CacheEntry newCacheEntry{columnBlock, reinterpret_cast<std::uintptr_t>(newPtr),
                                 sizeToInsert, lruQueue_.end()};
        auto cacheMapIt = cacheMap_.insert(std::make_pair(columnBlock, std::move(newCacheEntry))).first;
        lruQueue_.emplace_back(cacheMapIt->second);
        cacheMapIt->second.lruQueueIt = (--lruQueue_.end());
        return {newPtr, size, false};
    }
    /// <summary>
    /// Try to evict least recently used block, taking lock list into consideration
    /// </summary>
    /// <returns>True if something was evicted otherwise false</returns>
    bool Evict();
    /// <summary>
    /// Remove specific block from cache
    /// </summary>
    /// <param name="databaseName">Database of cached column</param>
    /// <param name="tableAndColumnName">table and column name of cached column</param>
    /// <param name="blockIndex">Block index of cached column</param>
    void ClearCachedBlock(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex);

    /// <summary>
    /// Check if block is in cache
    /// </summary>
    /// <param name="databaseName">Database of cached column</param>
    /// <param name="tableAndColumnName">table and column name of cached column</param>
    /// <param name="blockIndex">Block index of cached column</param>
    /// <returns>True if block is cached, otherwise false</returns>
    bool ContainsColumn(const std::string& databaseName,
                        const std::string& tableAndColumnName,
                        int32_t blockIndex,
                        int64_t loadSize,
                        int64_t loadOffset);


    void SetCurrentBlockIndex(size_t blockIndex)
    {
        currentBlockIndex_ = blockIndex;
    }
    GPUMemoryCache(const GPUMemoryCache&) = delete;
    GPUMemoryCache& operator=(const GPUMemoryCache&) = delete;
};
