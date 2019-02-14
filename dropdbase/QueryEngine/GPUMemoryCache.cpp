#include "GPUMemoryCache.h"
#include <boost/log/trivial.hpp>
#include "Context.h"

GPUMemoryCache::GPUMemoryCache(int32_t deviceID) :
	usedSize(0), deviceID_(deviceID)
{
	BOOST_LOG_TRIVIAL(debug) << "Cache initialized for device " << deviceID;
}

GPUMemoryCache::~GPUMemoryCache()
{
	lruQueue.clear();
	for (auto& cacheEntry : cacheMap)
	{
		Context::getInstance().getAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(cacheEntry.second.ptr), cacheEntry.second.size);
	}
	cacheMap.clear();
	BOOST_LOG_TRIVIAL(debug) << "~GPUMemoryCache" << deviceID_;
}

void GPUMemoryCache::evict()
{
	auto& front = lruQueue.front();
	BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << "Evict: " << reinterpret_cast<int8_t*>(front->second.ptr) << " " << front->second.size;
	Context::getInstance().getAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(front->second.ptr), front->second.size);
	usedSize -= front->second.size;
	BOOST_LOG_TRIVIAL(debug) << "GPUMemoryCache" << deviceID_ << "UsedSize: " << usedSize;
	cacheMap.erase(front);
	lruQueue.pop_front();
}

CudaMemAllocator & GPUMemoryCache::GetAllocator()
{
	return Context::getInstance().getAllocatorForDevice(deviceID_);
}

void GPUMemoryCache::clearCachedBlock(const std::string& columnName, int32_t blockIndex)
{
	std::string columnBlock = columnName + "_" + std::to_string(blockIndex);
	if (cacheMap.find(columnBlock) != cacheMap.end())
	{
		auto& toErase = cacheMap.at(columnBlock);
		lruQueue.erase(toErase.lruQueueIt);
		Context::getInstance().getAllocatorForDevice(deviceID_).deallocate(reinterpret_cast<int8_t*>(toErase.ptr), toErase.size);
		usedSize -= toErase.size;
		cacheMap.erase(cacheMap.find(columnBlock));
	}
	BOOST_LOG_TRIVIAL(debug) << "Cleared cached block "<< columnBlock << " on device" << deviceID_;
}

bool GPUMemoryCache::containsColumn(const std::string& columnName, int32_t blockIndex)
{
	std::string columnBlock = columnName + "_" + std::to_string(blockIndex);
	return cacheMap.find(columnBlock) != cacheMap.end();
}