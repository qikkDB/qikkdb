#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
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
											int32_t* currentBlockHisto, 
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

class GPUJoin
{
private:
    int32_t joinTableBlockCount_;
    int32_t joinTableBlockSize_;
    int32_t joinTableTotalSize_;

	int32_t* joinTableHashHistoTable_;
    int32_t* joinTableHashPrefixSum_;
    int32_t* joinTableHashTableBuckets_;

public:
    GPUJoin(int32_t joinTableBlockCount, int32_t joinTableBlockSize) : 
		joinTableBlockCount_(joinTableBlockCount), 
		joinTableBlockSize_(joinTableBlockSize), 
		joinTableTotalSize_(joinTableBlockCount * joinTableBlockSize)
    {
        GPUMemory::alloc(&joinTableHashHistoTable_, joinTableTotalSize_);
        GPUMemory::alloc(&joinTableHashPrefixSum_, joinTableTotalSize_);
        GPUMemory::alloc(&joinTableHashTableBuckets_, joinTableTotalSize_);
	}

	~GPUJoin()
    {
        GPUMemory::free(joinTableHashHistoTable_);
        GPUMemory::free(joinTableHashPrefixSum_);
        GPUMemory::free(joinTableHashTableBuckets_);
	}

	// Hash the first table into memory
	template<typename T>
	void HashBlock(T* tableBlock, int32_t blockOrder)
	{
		// Check if the chosen block is in range
		if (blockOrder < 0 || blockOrder >= joinTableBlockCount_)
		{
            std::cerr << "[ERROR] Block id out of limits during block hashing" << std::endl;
		}

		/////////////////////////////////////////////////////////////////////////////////////////
		// Calculate the histogram of hashes
		int32_t* currentBlockHisto = &joinTableHashHistoTable_[blockOrder * joinTableBlockSize_];
        kernel_calc_hash_histo<<<Context::getInstance().calcGridDim(joinTableBlockSize_),
                                 Context::getInstance().getBlockDim()>>>(currentBlockHisto, 
																		 tableBlock,
                                                                         joinTableBlockSize_);

		/////////////////////////////////////////////////////////////////////////////////////////
		// Calculate the prefix sum for this block
        int32_t* currentBlockPrefixSum = &joinTableHashPrefixSum_[blockOrder * joinTableBlockSize_];

        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        // Calculate the prefix sum
        // in-place scan
        cub::DeviceScan::ExclusiveSum(tempBuffer, tempBufferSize, currentBlockHisto,
                                      currentBlockPrefixSum, joinTableBlockSize_);
        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(tempBuffer, tempBufferSize, currentBlockHisto,
                                      currentBlockPrefixSum, joinTableBlockSize_);
        GPUMemory::free(tempBuffer);

		/////////////////////////////////////////////////////////////////////////////////////////
		// Save the data to buckets for later hash collision resolution
        int32_t* currentBlockBuckets = &joinTableHashTableBuckets_[blockOrder * joinTableBlockSize_];
		kernel_save_data_to_buckets<<<Context::getInstance().calcGridDim(joinTableBlockSize_),
                                      Context::getInstance().getBlockDim()>>>(currentBlockBuckets, 
																			  currentBlockHisto,
                                                                              currentBlockPrefixSum, 
																			  tableBlock,
                                                                              joinTableBlockSize_);
		// Restore the prefix sum buffer
		// IMPORTANT - the minus restores only those elements which have a nonzero occurance in the
		// histogram table !, but that is ok, because if an element has 0 occurances in this
		// table, it is not present and therefore no join is possible with this element
        GPUArithmetic::colCol<ArithmeticOperations::sub>(currentBlockPrefixSum,
                                                         currentBlockPrefixSum, 
														 currentBlockHisto, 
														 joinTableBlockSize_);
	}

	template <typename T>
	void JoinBlock(T* joinBlock)
	{

	}

	void debugInfo()
	{
        int32_t* host_joinTableHashHistoTable = new int32_t[joinTableTotalSize_];
        int32_t* host_joinTableHashPrefixSum = new int32_t[joinTableTotalSize_];
        int32_t* host_joinTableHashTableBuckets = new int32_t[joinTableTotalSize_];

		GPUMemory::copyDeviceToHost(host_joinTableHashHistoTable, joinTableHashHistoTable_, joinTableTotalSize_);
        GPUMemory::copyDeviceToHost(host_joinTableHashPrefixSum, joinTableHashPrefixSum_, joinTableTotalSize_);
        GPUMemory::copyDeviceToHost(host_joinTableHashTableBuckets, joinTableHashTableBuckets_, joinTableTotalSize_);

		for (int32_t i = 0; i < joinTableTotalSize_; i++)
		{
            std::printf("%d %d %d\n",
				host_joinTableHashHistoTable[i], host_joinTableHashPrefixSum[i], host_joinTableHashTableBuckets[i]);
		}
		
		delete[] host_joinTableHashHistoTable;
        delete[] host_joinTableHashPrefixSum;
        delete[] host_joinTableHashTableBuckets;

	}
};