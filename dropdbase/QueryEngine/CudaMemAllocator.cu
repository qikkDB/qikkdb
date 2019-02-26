#include "CudaMemAllocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <stdexcept>

CudaMemAllocator::CudaMemAllocator(int32_t deviceID) :
	deviceID_(deviceID)
{
	int32_t oldDevice;
	cudaGetDevice(&oldDevice);
	cudaSetDevice(deviceID);

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	//printf("Device %d: %s Total: %zu Free: %zu\n", deviceID_, props.name, total, free);
	if (cudaMalloc(&cudaBufferStart_, free - RESERVED_MEMORY) != cudaSuccess)
	{
		throw std::invalid_argument("Failed to alloc GPU buffer");
	}
	chainedBlocks_.push_back({ false, blocksBySize_.end(), free - RESERVED_MEMORY, cudaBufferStart_ });
	(*chainedBlocks_.begin()).sizeOrderIt = blocksBySize_.emplace(std::make_pair(free - RESERVED_MEMORY, chainedBlocks_.begin()));
#ifdef DEBUG_ALLOC
	logOut = fopen("E:\\alloc.log", "a");
	fprintf(logOut, "CudaMemAllocator %d\n", deviceID);
	fprintf(logOut, "Available blocks: %zu\n", chainedBlocks_.size());
	for (auto & ptrs : chainedBlocks_)
	{
		fprintf(logOut, "%zu bytes at %p\n", ptrs.blockSize, ptrs.ptr);
	}
#endif // DEBUG_ALLOC
	cudaSetDevice(oldDevice);
}

CudaMemAllocator::~CudaMemAllocator()
{
	if (cudaBufferStart_ != nullptr)
	{
		int oldDevice;
		cudaGetDevice(&oldDevice);
		cudaSetDevice(deviceID_);
		cudaFree(cudaBufferStart_);
		cudaSetDevice(oldDevice);
	}
#ifdef DEBUG_ALLOC
	if (logOut != nullptr)
	{
		fprintf(logOut, "~CudaMemAllocator %d\n", deviceID_);
		fclose(logOut);
	}
#endif // DEBUG_ALLOC
}

int8_t * CudaMemAllocator::allocate(std::ptrdiff_t numBytes)
{
	//Minimal allocation unit is 512bytes, same as cudaMalloc. Thurst relies on this internally.
	std::size_t alignedSize = numBytes % 512 == 0 ? numBytes : numBytes + (512 - numBytes % 512);
	auto it = blocksBySize_.lower_bound(alignedSize);
	if (it == blocksBySize_.end())
	{
		throw std::out_of_range("Out of GPU memory");
	}
	auto blockInfoIt = (*it).second;
	(*blockInfoIt).allocated = true;
	if ((*it).first > alignedSize)
	{
		SplitBlock(it, alignedSize);
	}
	blocksBySize_.erase(it);
	allocatedBlocks_.emplace(std::make_pair((*blockInfoIt).ptr, blockInfoIt));
#ifdef DEBUG_ALLOC
	fprintf(logOut,"%d CudaMemAllocator::allocate %p %zu\n", deviceID_, (*blockInfoIt).ptr, alignedSize);
	fflush(logOut);
#endif
	return static_cast<int8_t*>((*blockInfoIt).ptr);
}

void CudaMemAllocator::deallocate(int8_t * ptr, size_t numBytes)
{
#ifdef DEBUG_ALLOC
	fprintf(logOut, "%d CudaMemAllocator::deallocate ptr %p\n", deviceID_, ptr);
	fflush(logOut);
#endif
	auto allocListIt = allocatedBlocks_.find(ptr);
	if (allocListIt == allocatedBlocks_.end())
	{
		return;
	}
	auto listIt = (*allocListIt).second;
	allocatedBlocks_.erase(allocListIt);
	if (listIt != chainedBlocks_.begin())
	{
		auto prevIt = (listIt);
		prevIt--;
		if (!(*prevIt).allocated)
		{
			blocksBySize_.erase((*prevIt).sizeOrderIt);
			(*prevIt).blockSize += (*listIt).blockSize;
			chainedBlocks_.erase(listIt);
			listIt = prevIt;
		}
	}
	auto nextIt = listIt;
	nextIt++;
	if (nextIt != chainedBlocks_.end() && !(*nextIt).allocated)
	{
		blocksBySize_.erase((*nextIt).sizeOrderIt);
		(*listIt).blockSize += (*nextIt).blockSize;
		chainedBlocks_.erase(nextIt);
	}
	(*listIt).allocated = false;
	(*listIt).sizeOrderIt = blocksBySize_.emplace(std::make_pair((*listIt).blockSize, listIt));
#ifdef DEBUG_ALLOC
	fprintf(logOut, "CudaMemAllocator::deallocate final ptr %p %zu\n", (*listIt).ptr, (*listIt).blockSize);
	fflush(logOut);
#endif
}

void CudaMemAllocator::SplitBlock(std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator blockIterator, size_t requestedSize)
{
	auto blockInfoIt = (*blockIterator).second;
	size_t oldSize = (*blockIterator).first;
	void* newFreePtr = static_cast<int8_t*>((*blockInfoIt).ptr) + requestedSize;
	auto nextBlockInfo = blockInfoIt;
	nextBlockInfo++;
	auto listIt = chainedBlocks_.insert(nextBlockInfo, { false, blocksBySize_.end(), oldSize - requestedSize, newFreePtr });
	(*listIt).sizeOrderIt = blocksBySize_.emplace(std::make_pair(oldSize - requestedSize, listIt));
	(*blockInfoIt).blockSize = requestedSize;
}

void CudaMemAllocator::Clear()
{
#ifdef DEBUG_ALLOC
	fprintf(logOut, "---------------\nAllocation statistics for GPU %d:\n",deviceID_);
	fprintf(logOut, "Leaked pointers: %zu\n", allocatedBlocks_.size());
	for (auto & ptrs : allocatedBlocks_)
	{
		fprintf(logOut, "%zu bytes at %p\n", (*ptrs.second).blockSize, ptrs.first);
	}
	fprintf(logOut, "---------------\n");
	fprintf(logOut, "Available blocks: %zu\n", chainedBlocks_.size());
	for (auto & ptrs : chainedBlocks_)
	{
		fprintf(logOut, "%zu bytes at %p\n", ptrs.blockSize, ptrs.ptr);
	}
	fprintf(logOut, "---------------\n");
	fflush(logOut);
#endif
}