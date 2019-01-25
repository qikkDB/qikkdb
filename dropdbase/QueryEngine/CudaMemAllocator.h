#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <list>
#include <cstdio>

class CudaMemAllocator
{
private:
	struct BlockInfo 
	{
		bool allocated;
		std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator sizeOrderIt;
		size_t blockSize;
		void* ptr;
	};
#ifdef DEBUG_ALLOC
	FILE* logOut;
#endif // DEBUG_ALLOC

	void* cudaBufferStart_;
	CudaMemAllocator();
	std::list<BlockInfo> chainedBlocks_;
	std::multimap<size_t, std::list<BlockInfo>::iterator> blocksBySize_;
	std::unordered_map<void*, std::list<BlockInfo>::iterator> allocatedBlocks_;
	void SplitBlock(std::multimap<size_t, std::list<BlockInfo>::iterator>::iterator blockIterator, size_t requestedSize);
public:
	typedef int8_t value_type;
	static CudaMemAllocator& GetInstance();
	~CudaMemAllocator();
	CudaMemAllocator(const CudaMemAllocator&) = delete;
	CudaMemAllocator(const CudaMemAllocator&&) = delete;
	CudaMemAllocator& operator=(const CudaMemAllocator&) = delete;
	int8_t* allocate(std::ptrdiff_t num_bytes);
	void deallocate(int8_t* ptr, size_t num_bytes);
	void Clear();
};

