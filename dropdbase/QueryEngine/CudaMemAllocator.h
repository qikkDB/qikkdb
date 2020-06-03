#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <list>
#include <mutex>


/// A class for memory allocation operations
class CudaMemAllocator
{
private:
    /// Reserved memory for the GPU, allocating the entire GPU is forbidden
    static const int32_t RESERVED_MEMORY = 256000000;

    struct BlockInfo
    {
        bool allocated;
        std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator sizeOrderIt;
        size_t blockSize;
        void* ptr;
    };
    int32_t deviceID_;
#ifdef DEBUG_ALLOC
    FILE* logOut;
#endif // DEBUG_ALLOC

    void* cudaBufferStart_;
    std::list<BlockInfo> chainedBlocks_;
    std::multimap<size_t, std::list<BlockInfo>::iterator> blocksBySize_;
    std::unordered_map<void*, std::list<BlockInfo>::iterator> allocatedBlocks_;
    std::mutex allocator_mutex_; // Mutex for each allocator (gpu card)

    void SplitBlock(std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator blockIterator,
                    size_t requestedSize);
    friend void
    ValidateIterator(int gpuId,
                     const std::multimap<size_t, std::list<CudaMemAllocator::BlockInfo>::iterator>& container,
                     std::multimap<size_t, std::list<CudaMemAllocator::BlockInfo>::iterator>::iterator& iterator);
    friend void ValidateIterator(int gpuId,
                                 const std::list<CudaMemAllocator::BlockInfo>& container,
                                 std::list<CudaMemAllocator::BlockInfo>::iterator& iterator);
    friend void
    ValidateIterator(int gpuId,
                     const std::unordered_map<void*, std::list<CudaMemAllocator::BlockInfo>::iterator>& container,
                     std::unordered_map<void*, std::list<CudaMemAllocator::BlockInfo>::iterator>::iterator& iterator);

public:
    typedef int8_t value_type;
    CudaMemAllocator(int deviceID);
    ~CudaMemAllocator();
    CudaMemAllocator(const CudaMemAllocator&) = delete;
    CudaMemAllocator(CudaMemAllocator&& other) = delete;
    CudaMemAllocator& operator=(const CudaMemAllocator&) = delete;
    int8_t* Allocate(std::ptrdiff_t numBytes);
    void Deallocate(int8_t* ptr);
    void Clear();
};
