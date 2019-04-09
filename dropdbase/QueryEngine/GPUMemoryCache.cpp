#include "GPUMemoryCache.h"
#include <boost/log/trivial.hpp>
#include "Context.h"

std::vector<std::string> GPUMemoryCache::lockList;

void GPUMemoryCache::SetLockList(const std::vector<std::string>& lockList)
{
	GPUMemoryCache::lockList = lockList;
}

GPUMemoryCache::GPUMemoryCache(int32_t deviceID, size_t maximumSize) :
	usedSize(0), deviceID_(deviceID), maxSize_(maximumSize)
{
	BOOST_LOG_TRIVIAL(debug) << "Cache initialized for device " << deviceID;
}

GPUMemoryCache::~GPUMemoryCache()
{
	lruQueue.clear();
	for (auto& cacheEntry : cacheMap)
	{
		Context::getInstance().GetAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(cacheEntry.second.ptr), cacheEntry.second.size);
	}
	cacheMap.clear();
	BOOST_LOG_TRIVIAL(debug) << "~GPUMemoryCache" << deviceID_;
}

bool GPUMemoryCache::evict()
{
	for(auto it = lruQueue.begin(); it != lruQueue.end(); it++)
	{
		auto& queueItem = *it;
		bool isLockedItem = false;
		for (const auto& lockedColumn : GPUMemoryCache::lockList)
		{
			BOOST_LOG_TRIVIAL(debug) << "CacheLock cmp: " << lockedColumn << " " << it->ref.key);
			if (it->ref.key.find_first_of(lockedColumn, 0) == 0)
			{
				isLockedItem = true;
				break;
			}
		}

		if (isLockedItem)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << "Evict: " << reinterpret_cast<int8_t*>(queueItem.ref.ptr) << " " << queueItem.ref.size;
		Context::getInstance().GetAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(queueItem.ref.ptr), queueItem.ref.size);
		usedSize -= queueItem.ref.size;
		BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << "UsedSize: " << usedSize;
		cacheMap.erase(queueItem.ref.key);
		lruQueue.erase(it);
		return true;
	}
	return false;
}

CudaMemAllocator & GPUMemoryCache::GetAllocator()
{
	return Context::getInstance().GetAllocatorForDevice(deviceID_);
}

void GPUMemoryCache::clearCachedBlock(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex)
{
	std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex);
	if (cacheMap.find(columnBlock) != cacheMap.end())
	{
		auto& toErase = cacheMap.at(columnBlock);
		lruQueue.erase(toErase.lruQueueIt);
		Context::getInstance().GetAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(toErase.ptr), toErase.size);
		usedSize -= toErase.size;
		cacheMap.erase(cacheMap.find(columnBlock));
	}
	BOOST_LOG_TRIVIAL(debug) << "Cleared cached block "<< columnBlock << " on device" << deviceID_;
}

bool GPUMemoryCache::containsColumn(const std::string& databaseName, const std::string& tableAndColumnName, int32_t blockIndex)
{
	std::string columnBlock = databaseName + "." + tableAndColumnName + "_" + std::to_string(blockIndex);
	return cacheMap.find(columnBlock) != cacheMap.end();
}