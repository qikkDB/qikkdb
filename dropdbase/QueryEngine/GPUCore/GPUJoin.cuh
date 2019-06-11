#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <cmath>
#include "../../../cub/cub.cuh"

#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"

// Kernel - Calculate the hashed value occurances - histogram of hashes
template <typename T>
__global__ void kernel_calc_hash_histo(int32_t* currentBlockHisto, T* tableBlock, int32_t blockSize)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < blockSize; i += stride)
    {
        int32_t hash = tableBlock[i] % blockSize;
        atomicAdd(&currentBlockHisto[hash], 1);
    }
}

// Kernel - save the keys to a list ( into buckets) for later hash checking to avoid conflicts
template <typename T>
__global__ void kernel_save_data_to_buckets(int32_t* currentBlockBuckets,
											int32_t* currentBlockPrefixSum,
                                            T* tableBlock,
                                            int32_t blockSize)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < blockSize; i += stride)
    {
        int32_t hash = tableBlock[i] % blockSize;
        int32_t bucket_index = atomicAdd(&currentBlockPrefixSum[hash], 1);
        currentBlockBuckets[bucket_index] = tableBlock[i];
    }
}

// TODO - MAKE IT WORK
// Join the incoming block to the existing hash table
template <typename T>
__global__ void kernel_join_block_on_hash_table()
{

}

class GPUJoin
{
private:
    int32_t hashTableBlockCount_;
    int32_t hashTableBlockSize_;
    int32_t hashTableTotalSize_;

	int32_t* hashHistoTable_;
    int32_t* hashPrefixSumTable_;
    int32_t* hashBucketsTable_;

public:
    GPUJoin(int32_t hashTableBlockCount, int32_t hashTableBlockSize, int32_t resultTablePageSize)
    : 
		hashTableBlockCount_(hashTableBlockCount), 
		hashTableBlockSize_(hashTableBlockSize), 
		hashTableTotalSize_(hashTableBlockCount * hashTableBlockSize)
    {
		// Alloc buffers
        GPUMemory::alloc(&hashHistoTable_, hashTableTotalSize_);
        GPUMemory::alloc(&hashPrefixSumTable_, hashTableTotalSize_);
        GPUMemory::alloc(&hashBucketsTable_, hashTableTotalSize_);
	}

	~GPUJoin()
    {
        GPUMemory::free(hashHistoTable_);
        GPUMemory::free(hashPrefixSumTable_);
        GPUMemory::free(hashBucketsTable_);
	}

	// Hash the first table into memory
	template<typename T>
	void HashBlock(T* tableBlock, int32_t blockOrder)
	{
		// Check if the chosen block is in range
		if (blockOrder < 0 || blockOrder >= hashTableBlockCount_)
		{
            std::cerr << "[ERROR] Block id out of limits during block hashing" << std::endl;
		}

		/////////////////////////////////////////////////////////////////////////////////////////
		// Calculate the histogram of hashes
		int32_t* currentBlockHisto = &hashHistoTable_[blockOrder * hashTableBlockSize_];
        kernel_calc_hash_histo<<<Context::getInstance().calcGridDim(hashTableBlockSize_),
                                 Context::getInstance().getBlockDim()>>>(currentBlockHisto, 
																		 tableBlock,
                                                                         hashTableBlockSize_);

		/////////////////////////////////////////////////////////////////////////////////////////
		// Calculate the prefix sum for this block
        int32_t* currentBlockPrefixSum = &hashPrefixSumTable_[blockOrder * hashTableBlockSize_];

        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        // Calculate the prefix sum
        // in-place scan
        cub::DeviceScan::ExclusiveSum(tempBuffer, tempBufferSize, currentBlockHisto,
                                      currentBlockPrefixSum, hashTableBlockSize_);
        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(tempBuffer, tempBufferSize, currentBlockHisto,
                                      currentBlockPrefixSum, hashTableBlockSize_);
        GPUMemory::free(tempBuffer);

		/////////////////////////////////////////////////////////////////////////////////////////
		// Save the data to buckets for later hash collision resolution
        int32_t* currentBlockBuckets = &hashBucketsTable_[blockOrder * hashTableBlockSize_];
		kernel_save_data_to_buckets<<<Context::getInstance().calcGridDim(hashTableBlockSize_),
                                      Context::getInstance().getBlockDim()>>>(currentBlockBuckets, 
                                                                              currentBlockPrefixSum, 
																			  tableBlock,
                                                                              hashTableBlockSize_);
		// Restore the prefix sum buffer
		// IMPORTANT - the minus restores only those elements which have a nonzero occurance in the
		// histogram table !, but that is ok, because if an element has 0 occurances in this
		// table, it is not present and therefore no join is possible with this element
        GPUArithmetic::colCol<ArithmeticOperations::sub>(currentBlockPrefixSum,
                                                         currentBlockPrefixSum, 
														 currentBlockHisto, 
														 hashTableBlockSize_);
	}

	template <typename T>
    void JoinBlockOnHashTable(T* joinBlock, int32_t joinBlockSize)
	{
        // TODO - MAKE IT WORK
	}

	void debugInfo()
	{
        int32_t* host_joinTableHashHistoTable = new int32_t[hashTableTotalSize_];
        int32_t* host_joinTableHashPrefixSum = new int32_t[hashTableTotalSize_];
        int32_t* host_joinTableHashTableBuckets = new int32_t[hashTableTotalSize_];

		GPUMemory::copyDeviceToHost(host_joinTableHashHistoTable, hashHistoTable_, hashTableTotalSize_);
        GPUMemory::copyDeviceToHost(host_joinTableHashPrefixSum, hashPrefixSumTable_, hashTableTotalSize_);
        GPUMemory::copyDeviceToHost(host_joinTableHashTableBuckets, hashBucketsTable_, hashTableTotalSize_);

		for (int32_t i = 0; i < hashTableTotalSize_; i++)
		{
            std::printf("%d %d %d\n",
				host_joinTableHashHistoTable[i], host_joinTableHashPrefixSum[i], host_joinTableHashTableBuckets[i]);
		}
		
		delete[] host_joinTableHashHistoTable;
        delete[] host_joinTableHashPrefixSum;
        delete[] host_joinTableHashTableBuckets;
	}
};