#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <vector>

#include "../Context.h"
#include "GPUMemory.cuh"

__device__ const int32_t HASH_BUCKET_COUNT = 1024;

template <typename T>
__global__ void kernel_calc_histo(T* RCol, int32_t* hashTableHisto, int32_t dataElementCount)
{
    __shared__ int32_t shared_mem[HASH_BUCKET_COUNT];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
    }
}

class GPUJoin
{
public:
    // Constructor, requires the number of GPU blocks and their size
    GPUJoin(int32_t hashTableBlockCount, int32_t hashTableBlockDataElementCount)
    : hashTableBlockCount_(hashTableBlockCount),
      hashTableBlockDataElementCount_(hashTableBlockDataElementCount),
      hashTableHisto_(hashTableBlockCount), hashTablePrefixSum_(hashTableBlockCount),
      hashTableId_(hashTableBlockCount)
    {
        // Alloc the hash table for each block
        for (int32_t i = 0; i < hashTableBlockCount_; i++)
        {
            GPUMemory::alloc(&hashTableHisto_[i], hashTableBlockDataElementCount_);
            GPUMemory::alloc(&hashTablePrefixSum_[i], hashTableBlockDataElementCount_);
            GPUMemory::alloc(&hashTableId_[i], hashTableBlockDataElementCount_);
        }
    }

    ~GPUJoin()
    {
        // Dealloc the hash table for each block
        for (int32_t i = 0; i < hashTableBlockCount_; i++)
        {
            GPUMemory::free(hashTableHisto_[i]);
            GPUMemory::free(hashTablePrefixSum_[i]);
            GPUMemory::free(hashTableId_[i]);
        }
    }

    // Construct a hash table histogram and prefix sum per block and keep it on the GPU
    template <typename T>
    void HashBlock(T* RCol, int32_t blockOrder, int32_t dataElementCount)
    {
        int32_t* processed_block = hashTableHisto_[blockOrder];

        kernel_calc_histo<<<Context::getInstance().calcGridDim(dataElementCount),
                            Context::getInstance().getBlockDim()>>>(RCol, processed_block, dataElementCount);
    }

	template <typename T>
    void JoinBlock(T* SCol, int32_t dataElementCount)
    {

    }

private:
    int32_t hashTableBlockCount_;
    int32_t hashTableBlockDataElementCount_;

    std::vector<int32_t*> hashTableHisto_;
    std::vector<int32_t*> hashTablePrefixSum_;
    std::vector<int32_t*> hashTableId_;
};