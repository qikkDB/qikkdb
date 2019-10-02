#include "CudaMemAllocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef DEBUG_ALLOC
#include <cstdio>
#include <iostream>
#ifndef WIN32
#include <execinfo.h>
#endif
#endif
#include <stdexcept>

#ifdef DEBUG_ALLOC
void ValidateIterator(int gpuId,
                      const std::multimap<size_t, std::list<CudaMemAllocator::BlockInfo>::iterator>& container,
                      std::multimap<size_t, std::list<CudaMemAllocator::BlockInfo>::iterator>::iterator& iterator)
{
    if (iterator == container.end())
    {
        return;
    }
    for (auto it = container.begin(); it != container.end(); it++)
    {
        if (it == iterator)
        {
            return;
        }
    }
    std::cerr << "ASSERTION FAILURE: invalid iterator to Allocator multimap for GPU " << gpuId << "\nBacktrace:\n";
#ifndef WIN32
    void* backtraceArray[25];
    int btSize = backtrace(backtraceArray, 25);
    char** symbols = backtrace_symbols(backtraceArray, btSize);
    for (int i = 0; i < btSize; i++)
    {
        std::cerr << i << ": " << symbols[i] << "\n";
    }
    free(symbols);
    std::cerr << "-- Backtrace end --" << std::endl;
#endif
    abort();
}

void ValidateIterator(int gpuId,
                      const std::list<CudaMemAllocator::BlockInfo>& container,
                      std::list<CudaMemAllocator::BlockInfo>::iterator& iterator)
{
    if (iterator == container.end())
    {
        return;
    }
    for (auto it = container.begin(); it != container.end(); it++)
    {
        if (it == iterator)
        {
            return;
        }
    }
    std::cerr << "ASSERTION FAILURE: invalid iterator to Allocator BlockList for GPU " << gpuId << "\nBacktrace:\n";
#ifndef WIN32
    void* backtraceArray[25];
    int btSize = backtrace(backtraceArray, 25);
    char** symbols = backtrace_symbols(backtraceArray, btSize);
    for (int i = 0; i < btSize; i++)
    {
        std::cerr << i << ": " << symbols[i] << "\n";
    }
    free(symbols);
    std::cerr << "-- Backtrace end --" << std::endl;
#endif
    abort();
}

void ValidateIterator(int gpuId,
                      const std::unordered_map<void*, std::list<CudaMemAllocator::BlockInfo>::iterator>& container,
                      std::unordered_map<void*, std::list<CudaMemAllocator::BlockInfo>::iterator>::iterator& iterator)
{
    if (iterator == container.end())
    {
        return;
    }
    for (auto it = container.begin(); it != container.end(); it++)
    {
        if (it == iterator)
        {
            return;
        }
    }
    std::cerr << "ASSERTION FAILURE: invalid iterator to Allocator pointer map for GPU " << gpuId << "\nBacktrace:\n";
#ifndef WIN32
    void* backtraceArray[25];
    int btSize = backtrace(backtraceArray, 25);
    char** symbols = backtrace_symbols(backtraceArray, btSize);
    for (int i = 0; i < btSize; i++)
    {
        std::cerr << i << ": " << symbols[i] << "\n";
    }
    free(symbols);
    std::cerr << "-- Backtrace end --" << std::endl;
#endif
    abort();
}
#endif

/// Initiaize the neccessary data structures for the GPUallocator
CudaMemAllocator::CudaMemAllocator(int32_t deviceID) : deviceID_(deviceID)
{
    int32_t oldDevice;
    cudaGetDevice(&oldDevice);
    cudaSetDevice(deviceID);

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    // printf("Device %d: %s Total: %zu Free: %zu\n", deviceID_, props.name, total, free);
    if (cudaMalloc(&cudaBufferStart_, free - RESERVED_MEMORY) != cudaSuccess)
    {
        throw std::invalid_argument("Failed to alloc GPU buffer");
    }
    chainedBlocks_.push_back({false, blocksBySize_.end(), free - RESERVED_MEMORY, cudaBufferStart_});
    (*chainedBlocks_.begin()).sizeOrderIt =
        blocksBySize_.emplace(std::make_pair(free - RESERVED_MEMORY, chainedBlocks_.begin()));
#ifdef DEBUG_ALLOC
    logOut = fopen("C:\dbg-alloc.log", "a");
    fprintf(logOut, "CudaMemAllocator %d\n", deviceID);
    fprintf(logOut, "Available blocks: %zu\n", chainedBlocks_.size());
    for (auto& ptrs : chainedBlocks_)
    {
        fprintf(logOut, "%zu bytes at %p\n", ptrs.blockSize, ptrs.ptr);
    }
#endif // DEBUG_ALLOC
    cudaSetDevice(oldDevice);
}

/// Destroy the allocator
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

/// Allocate data with the allocator
/// < param name="numBytes">number of bytes to be allocated like size_t</param>
/// <returns> a chunk of allocated memory on the GPU</returns>
int8_t* CudaMemAllocator::allocate(std::ptrdiff_t numBytes)
{
    if (numBytes <= 0)
    {
        throw std::out_of_range("Invalid allocation size");
    }
#ifdef DEBUG_ALLOC
    fprintf(logOut, "%d CudaMemAllocator::allocate %zu bytes\n", deviceID_, numBytes);
    fprintf(logOut, "-- Backtrace start --\n");
#ifndef WIN32
    void* backtraceArray[25];
    int btSize = backtrace(backtraceArray, 25);
    char** symbols = backtrace_symbols(backtraceArray, btSize);
    for (int i = 0; i < btSize; i++)
    {
        fprintf(logOut, "%d: %s\n", i, symbols[i]);
    }
    free(symbols);
    fprintf(logOut, "-- Backtrace end --\n");
#endif
    fflush(logOut);
#endif
    // Minimal allocation unit is 512bytes, same as cudaMalloc. Thurst relies on this internally.
    std::size_t alignedSize = numBytes % 512 == 0 ? numBytes : numBytes + (512 - numBytes % 512);
    auto it = blocksBySize_.lower_bound(alignedSize);
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, blocksBySize_, it);
#endif
    if (it == blocksBySize_.end())
    {
#ifdef DEBUG_ALLOC
        fprintf(logOut, "Out of GPU memory\n");
#endif
        throw std::out_of_range("Out of GPU memory");
    }
    auto blockInfoIt = (*it).second;
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, blockInfoIt);
#endif
    (*blockInfoIt).allocated = true;
    if ((*it).first > alignedSize)
    {
        SplitBlock(it, alignedSize);
    }
    blocksBySize_.erase(it);
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, blockInfoIt);
#endif
    allocatedBlocks_.emplace(std::make_pair((*blockInfoIt).ptr, blockInfoIt));
#ifdef DEBUG_ALLOC
    fprintf(logOut, "%d                -allocated %p %zu\n", deviceID_, (*blockInfoIt).ptr, alignedSize);
    fflush(logOut);
#endif
    return static_cast<int8_t*>((*blockInfoIt).ptr);
}

/// Deallocate data with the allocator
/// < param name="ptr">the pointer to be freed</param>
/// < param name="numBytes">number of byte to free</param>
void CudaMemAllocator::deallocate(int8_t* ptr, size_t numBytes)
{
#ifdef DEBUG_ALLOC
    fprintf(logOut, "%d CudaMemAllocator::deallocate ptr %p\n", deviceID_, ptr);
#ifndef WIN32
    fprintf(logOut, "-- Backtrace start --\n");
    void* backtraceArray[25];
    int btSize = backtrace(backtraceArray, 25);
    char** symbols = backtrace_symbols(backtraceArray, btSize);
    for (int i = 0; i < btSize; i++)
    {
        fprintf(logOut, "%d: %s\n", i, symbols[i]);
    }
    free(symbols);
    fprintf(logOut, "-- Backtrace end --\n");
#endif
    fflush(logOut);
#endif
    auto allocListIt = allocatedBlocks_.find(ptr);
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, allocatedBlocks_, allocListIt);
#endif
    if (allocListIt == allocatedBlocks_.end())
    {
        throw std::out_of_range("Attempted to free unallocated pointer");
    }
    auto listIt = (*allocListIt).second;
    allocatedBlocks_.erase(allocListIt);
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, listIt);
#endif
    if (listIt != chainedBlocks_.begin())
    {
        auto prevIt = (listIt);
        prevIt--;
#ifdef DEBUG_ALLOC
        ValidateIterator(deviceID_, chainedBlocks_, prevIt);
#endif
        if (!(*prevIt).allocated)
        {
#ifdef DEBUG_ALLOC
            ValidateIterator(deviceID_, blocksBySize_, (*prevIt).sizeOrderIt);
#endif
            blocksBySize_.erase((*prevIt).sizeOrderIt);
            (*prevIt).blockSize += (*listIt).blockSize;
            chainedBlocks_.erase(listIt);
            listIt = prevIt;
        }
    }
    auto nextIt = listIt;
    nextIt++;
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, nextIt);
#endif
    if (nextIt != chainedBlocks_.end() && !(*nextIt).allocated)
    {
        blocksBySize_.erase((*nextIt).sizeOrderIt);
        (*listIt).blockSize += (*nextIt).blockSize;
        chainedBlocks_.erase(nextIt);
    }
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, listIt);
#endif
    (*listIt).allocated = false;
    (*listIt).sizeOrderIt = blocksBySize_.emplace(std::make_pair((*listIt).blockSize, listIt));
#ifdef DEBUG_ALLOC
    fprintf(logOut, "CudaMemAllocator::deallocate final ptr %p %zu\n", (*listIt).ptr, (*listIt).blockSize);
    fflush(logOut);
#endif
}

void CudaMemAllocator::SplitBlock(std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator blockIterator,
                                  size_t requestedSize)
{
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, blocksBySize_, blockIterator);
#endif
    auto blockInfoIt = (*blockIterator).second;
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, blockInfoIt);
#endif
    size_t oldSize = (*blockIterator).first;
    void* newFreePtr = static_cast<int8_t*>((*blockInfoIt).ptr) + requestedSize;
    auto nextBlockInfo = blockInfoIt;
    nextBlockInfo++;
#ifdef DEBUG_ALLOC
    ValidateIterator(deviceID_, chainedBlocks_, nextBlockInfo);
#endif
    auto listIt = chainedBlocks_.insert(nextBlockInfo, {false, blocksBySize_.end(),
                                                        oldSize - requestedSize, newFreePtr});
    (*listIt).sizeOrderIt = blocksBySize_.emplace(std::make_pair(oldSize - requestedSize, listIt));
    (*blockInfoIt).blockSize = requestedSize;
}

/// Clear the allocator's contents - free all memory
void CudaMemAllocator::Clear()
{
#ifdef DEBUG_ALLOC
    fprintf(logOut, "---------------\nAllocation statistics for GPU %d:\n", deviceID_);
    fprintf(logOut, "Leaked pointers: %zu\n", allocatedBlocks_.size());
    for (auto& ptrs : allocatedBlocks_)
    {
        fprintf(logOut, "%zu bytes at %p\n", (*ptrs.second).blockSize, ptrs.first);
    }
    fprintf(logOut, "---------------\n");
    fprintf(logOut, "Available blocks: %zu\n", chainedBlocks_.size());
    for (auto& ptrs : chainedBlocks_)
    {
        fprintf(logOut, "%zu bytes at %p\n", ptrs.blockSize, ptrs.ptr);
    }
    fprintf(logOut, "---------------\n");
    fflush(logOut);
#endif
}
