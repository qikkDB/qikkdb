#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <list>

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
    void SplitBlock(std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator blockIterator,
                    size_t requestedSize);

public:
    typedef int8_t value_type;
    CudaMemAllocator(int deviceID);
    ~CudaMemAllocator();
    CudaMemAllocator(const CudaMemAllocator&) = delete;
    CudaMemAllocator(CudaMemAllocator&& other) = delete;
    CudaMemAllocator& operator=(const CudaMemAllocator&) = delete;
    int8_t* allocate(std::ptrdiff_t num_bytes);
    void deallocate(int8_t* ptr, size_t num_bytes);
    void Clear();
};
