#include "GPUMemoryCache.h"
#include "GPUMemoryCache.h"
#include <boost/log/trivial.hpp>
#include "Context.h"

std::vector<std::string> GPUMemoryCache::lockList;

void GPUMemoryCache::SetLockList(const std::vector<std::string>& lockList)
{
    GPUMemoryCache::lockList = lockList;
}

GPUMemoryCache::GPUMemoryCache(int32_t deviceID, size_t maximumSize)
: maxSize_(maximumSize), deviceID_(deviceID), usedSize(0), currentBlockIndex_(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Cache initialized for device " << deviceID;
}

GPUMemoryCache::~GPUMemoryCache()
{
    lruQueue.clear();
    for (auto& cacheEntry : cacheMap)
    {
        Context::getInstance().GetAllocatorForDevice(deviceID_).Deallocate(reinterpret_cast<int8_t*>(
                                                                               cacheEntry.second.ptr));
    }
    cacheMap.clear();
    BOOST_LOG_TRIVIAL(debug) << "~GPUMemoryCache" << deviceID_;
}

bool GPUMemoryCache::evict()
{
    for (auto it = lruQueue.begin(); it != lruQueue.end(); it++)
    {
        auto& queueItem = *it;
        bool isLockedItem = false;
        // Check if current eviction candidate is evictable
        for (const auto& lockedColumn : GPUMemoryCache::lockList)
        {
            std::string currentBlockIndexStr = std::to_string(currentBlockIndex_);
            if (it->ref.key.find(lockedColumn, 0) == 0 &&
                it->ref.key.find_last_of(currentBlockIndexStr) ==
                    lockedColumn.length() + 1)  // +1 for "_" before blockIndex
            {
                isLockedItem = true;
                break;
            }
        }

        if (isLockedItem)
        {
            BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << " Locked: " << it->ref.key;
            continue;
        }

        BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_
                                 << " Evict: " << queueItem.ref.key << " " << reinterpret_cast<int8_t*>(queueItem.ref.ptr) << " "
                                 << queueItem.ref.size;
        Context::getInstance().GetAllocatorForDevice(deviceID_).Deallocate(reinterpret_cast<int8_t*>(
                                                                               queueItem.ref.ptr));
        usedSize -= queueItem.ref.size;
        BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << " UsedSize: " << usedSize;
        cacheMap.erase(queueItem.ref.key);
        lruQueue.erase(it);
        return true;
    }
    return false;
}

std::map<std::string, GPUMemoryCache::CacheEntry>::iterator GPUMemoryCache::FindPrefix(const std::string& search_for)
{
    std::map<std::string, CacheEntry>::iterator i = cacheMap.lower_bound(search_for);
    if (i != cacheMap.end())
    {
        const std::string& key = i->first;
        if (key.compare(0, search_for.size(), search_for) == 0)
        {
            return i;
        }
    }
    return cacheMap.end();
}

CudaMemAllocator& GPUMemoryCache::GetAllocator()
{
    return Context::getInstance().GetAllocatorForDevice(deviceID_);
}

void GPUMemoryCache::clearCachedBlock(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex)
{
    std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex);

    std::map<std::string, GPUMemoryCache::CacheEntry>::iterator toErase = FindPrefix(columnBlock);
    while (toErase != cacheMap.end())
    {
        lruQueue.erase(toErase->second.lruQueueIt);
        Context::getInstance().GetAllocatorForDevice(deviceID_).Deallocate(reinterpret_cast<int8_t*>(
                                                                               toErase->second.ptr));
        usedSize -= toErase->second.size;
        cacheMap.erase(toErase);
        toErase = FindPrefix(columnBlock);
    }
    // BOOST_LOG_TRIVIAL(debug) << "Cleared cached block " << columnBlock << " on device" << deviceID_;
}

bool GPUMemoryCache::containsColumn(const std::string& databaseName,
                                    const std::string& tableAndColumnName,
                                    int32_t blockIndex,
                                    int64_t loadSize,
                                    int64_t loadOffset)
{
    std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex) +
                              "_" + std::to_string(loadSize) + "_" + std::to_string(loadOffset);
    return cacheMap.find(columnBlock) != cacheMap.end();
}
