#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <vector>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "../../../cub/cub.cuh"

__device__ const int32_t HASH_BUCKET_COUNT = 1024; // Context::getInstance().getBlockDim();

__device__ int32_t hash(int32_t key)
{
    return key % HASH_BUCKET_COUNT;
}

template <typename T>
__global__ void kernel_calc_histo(T* RCol, int32_t* hashTableHisto_, int32_t dataElementCount)
{
    // Shared memory for the local hash value histograms
    __shared__ int32_t shared_mem[HASH_BUCKET_COUNT];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Set all local memory locations to zero and wait for each other
        shared_mem[threadIdx.x] = 0;
        __syncthreads();

        // Hash the values of the input rows and atomically increment the counters in the local memory
        atomicAdd(&shared_mem[hash(RCol[i])], 1);

        // Wait for each other again and write the results, so the local memory is free for the next batch
        __syncthreads();
        hashTableHisto_[i] = shared_mem[threadIdx.x];
    }
}

/*
template <typename T>
__global__ void kernel_sort_local(T* RCol,
								  int32_t* hashTableHisto_,
                                  int32_t* hashTablePrefixSum_, 
								  int32_t* hashTableSortedId_, 
								  int32_t dataElementCount)
{
    // Shared memory for the local hash value histograms
    __shared__ int32_t shared_mem[HASH_BUCKET_COUNT];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Load the histograms into memory
        shared_mem[threadIdx.x] = hashTablePrefixSum_[i];
        __syncthreads();

        // Sort the id's of the entries based on the histogram
        for (int32_t j = 0; j < hashTableHisto_[i]; j++)
		{
            int32_t old_idx = atomicAdd(&shared_mem[threadIdx.x], 1);
            hashTableSortedId_[old_idx] = threadIdx.x;
		}
    }
}*/

class GPUJoin
{
public:
    // Constructor, requires the number of GPU blocks and their size
    GPUJoin(int32_t hashTableBlockCount, int32_t hashTableBlockSize)
    : hashTableBlockCount_(hashTableBlockCount), 
	  hashTableBlockSize_(hashTableBlockSize), 
	  hashTableTotalSize_(hashTableBlockCount_ * hashTableBlockSize_),
	  hashTableHisto_(nullptr),
      hashTablePrefixSum_(nullptr), 
	  hashTableSortedId_(nullptr)
    {
        // Alloc the hash table for each block
        GPUMemory::alloc(&hashTableHisto_, hashTableTotalSize_);
        GPUMemory::alloc(&hashTablePrefixSum_, hashTableTotalSize_);
        GPUMemory::alloc(&hashTableSortedId_, hashTableTotalSize_);
    }

    ~GPUJoin()
    {
        // Dealloc the hash table for each block
        GPUMemory::free(hashTableHisto_);
        GPUMemory::free(hashTablePrefixSum_);
        GPUMemory::free(hashTableSortedId_);
    }

    // Construct a hash table histogram and prefix sum per block and keep it on the GPU
    template <typename T>
    void CalcBlockHashHisto(T* RCol, int32_t blockOrder)
    {
        if (blockOrder < 0 || blockOrder >= hashTableBlockCount_)
        {
            std::cerr << "[ERROR] Hash data block order index out of range" << std::endl;
        }

		// Calculate the hash histograms
        int32_t* processed_block = &hashTableHisto_[blockOrder * hashTableBlockSize_];
        kernel_calc_histo<<<Context::getInstance().calcGridDim(hashTableBlockSize_),
                            Context::getInstance().getBlockDim()>>>(RCol, processed_block, hashTableBlockSize_);
    }

	void CalcGlobalPrefixSum()
    {
        // Calculate the exclusive prefix sum of the hash table histograms
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;

        // Calculate the prefix sum
        // in-place scan
        cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, hashTableHisto_, hashTablePrefixSum_,
                                      hashTableBlockCount_ * hashTableBlockSize_);
        
		// Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
        
		// Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(tempBuffer, tempBufferSize, hashTableHisto_, hashTablePrefixSum_,
                                      hashTableBlockCount_ * hashTableBlockSize_);
        GPUMemory::free(tempBuffer);
	}

	template <typename T>
	void SortHashTableIDs(T* RCol, int32_t blockOrder)
    {
        /*
		if (blockOrder < 0 || blockOrder >= hashTableBlockCount_)
        {
            std::cerr << "[ERROR] Hash data block order index out of range" << std::endl;
        }

		// Sort the row indices in blocks locally
        int32_t* processed_block_histo = &hashTableHisto_[blockOrder * hashTableBlockSize_];
        int32_t* processed_block_prefix_sum = &hashTablePrefixSum_[blockOrder * hashTableBlockSize_];
        int32_t* processed_block_sorted_id = &hashTableSortedId_[blockOrder * hashTableBlockSize_];

		kernel_sort_local<<<Context::getInstance().calcGridDim(hashTableBlockSize_),
                            Context::getInstance().getBlockDim()>>>(RCol, 
																	processed_block_histo,
																	processed_block_prefix_sum,
																	processed_block_sorted_id,
                                                                    hashTableBlockSize_);
																	*/
	}

	void PrintDebug()
    {
        // DEBUG - Print the intermediate results
        int32_t* hashTableHisto_data = new int32_t[hashTableBlockCount_ * hashTableBlockSize_];
        int32_t* hashTablePrefixSum_data = new int32_t[hashTableBlockCount_ * hashTableBlockSize_];
        int32_t* hashTableSortedId_data = new int32_t[hashTableBlockCount_ * hashTableBlockSize_];

        GPUMemory::copyDeviceToHost(hashTableHisto_data, hashTableHisto_, hashTableBlockCount_ * hashTableBlockSize_);
        GPUMemory::copyDeviceToHost(hashTablePrefixSum_data, hashTablePrefixSum_,
                                    hashTableBlockCount_ * hashTableBlockSize_);
        GPUMemory::copyDeviceToHost(hashTableSortedId_data, hashTableSortedId_,
                                    hashTableBlockCount_ * hashTableBlockSize_);

        for (int32_t i = 0; i < hashTableBlockCount_ * hashTableBlockSize_; i++)
        {
            std::printf("%5d %5d %5d %5d\n", i % hashTableBlockSize_, hashTableHisto_data[i],
                        hashTablePrefixSum_data[i], hashTableSortedId_data[i]);
        }

		delete[] hashTableHisto_data;
        delete[] hashTablePrefixSum_data;
        delete[] hashTableSortedId_data;
	}

    template <typename T>
    void JoinBlock(T* SCol, int32_t dataElementCount)
    {
    }

private:
    int32_t hashTableBlockCount_;
    int32_t hashTableBlockSize_;
    int32_t hashTableTotalSize_;

    int32_t* hashTableHisto_;
    int32_t* hashTablePrefixSum_;
    int32_t* hashTableSortedId_;
};