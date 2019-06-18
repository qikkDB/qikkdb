#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <iostream>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"

#include "../../../cub/cub.cuh"

__device__ const int32_t HASH_TABLE_SUB_SIZE = 0x400;
__device__ const int32_t HASH_MOD = 0x3FF;

__device__ int32_t hash(int32_t key)
{
    return key & HASH_MOD;
}

template<typename T>
__global__ void kernel_calc_hash_histo(int32_t* HashTableHisto,
									   int32_t hashTableSize,
									   T* RTable, 
									   int32_t dataElementCount)
{
    __shared__ int32_t shared_memory[HASH_TABLE_SUB_SIZE];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < hashTableSize && i < dataElementCount; i += stride)
    {
		// Count the occurances of hashes and accumulate them to the local memory
        shared_memory[threadIdx.x] = 0;
        __syncthreads();

		int32_t hash_idx = hash(RTable[i]);
        atomicAdd(&shared_memory[hash_idx], 1);

		__syncthreads();
        HashTableHisto[i] = shared_memory[threadIdx.x];
    }
}

template <typename T>
__global__ void kernel_put_data_to_buckets(int32_t* HashTableHashBuckets,
										   int32_t* HashTablePrefixSum,
										   int32_t hashTableSize,
                                           T* RTable,
										   int32_t dataElementCount)
{
    __shared__ int32_t shared_memory[HASH_TABLE_SUB_SIZE];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < hashTableSize && i < dataElementCount; i += stride)
    {
        shared_memory[threadIdx.x] = (i == 0) ? 0 : HashTablePrefixSum[i - 1];
        __syncthreads();

        int32_t hash_idx = hash(RTable[i]);
		int32_t bucket_idx = atomicAdd(&shared_memory[hash_idx], 1);
		HashTableHashBuckets[bucket_idx] = i;//RTable[i];
    }
}

template <typename T>
__global__ void kernel_calc_join_histo(int32_t* JoinTableHisto,
									   int32_t joinTableSize,
									   int32_t* HashTableHisto, 
									   int32_t* HashTablePrefixSum,
                                       int32_t* HashTableHashBuckets,
									   int32_t hashTableSize,
									   T* RTable, 
									   int32_t dataElementCountRTable,
									   T* STable,
									   int32_t dataElementCountSTable)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < joinTableSize && i < dataElementCountSTable; i += stride)
    {
		// Zero the histo array
		JoinTableHisto[i] = 0;

		// Count the number of result hash matches for this entry
		int32_t hashMatchCounter = 0;
        
		// Hash table buckets probing and occurance counting
        int32_t hash_idx = hash(STable[i]);
        for (int32_t j = hash_idx; j < hashTableSize; j += HASH_TABLE_SUB_SIZE)
		{
			// Check if a bucket is empty, if yes, try the next bucket with the same hash
			// Otherwise probe and count the number of matching entries
            for (int32_t k = 0; k < HashTableHisto[j]; k++)
            {
                if (RTable[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]] == STable[i])
				{
                    hashMatchCounter++;
				}
			}
		}
        JoinTableHisto[i] = hashMatchCounter;
    }
}

template <typename T>
__global__ void kernel_distribute_results_to_buffer(T* QTableA,
                                                    T* QTableB,
													int32_t* JoinTableHisto,
													int32_t* JoinTablePrefixSum,
													int32_t joinTableSize,
                                                    int32_t* HashTableHisto,
                                                    int32_t* HashTablePrefixSum,
                                                    int32_t* HashTableHashBuckets,
													int32_t hashTableSize,
													T* RTable,
													int32_t dataElementCountRTable,
                                                    T* STable,
													int32_t dataElementCountSTable)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < joinTableSize && i < dataElementCountSTable; i += stride)
    {
		int32_t join_prefix_sum_offset_index = 0;
		// Hash table buckets probing
		int32_t hash_idx = hash(STable[i]);
		for (int32_t j = hash_idx; j < hashTableSize; j += HASH_TABLE_SUB_SIZE)
		{
			// Check if a bucket is empty, if yes, try the next bucket with the same hash
			// Otherwise write count the number of matching entries
			for (int32_t k = 0; k < HashTableHisto[j]; k++)
			{
				// Write the results if the value in a bucket matches the present value
				// Write them to the calculated offset of the prefix sum buffer
				if (join_prefix_sum_offset_index < JoinTableHisto[i] && 
					RTable[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]] == STable[i])
				{
					QTableA[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] = HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k];
					QTableB[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] = i;//STable[i];
					join_prefix_sum_offset_index++;
				}
			}
		}
	}
}

class GPUJoin
{
private:
	int32_t hashTableSize_;
	int32_t joinTableSize_;

    int32_t* HashTableHisto_;
    int32_t* HashTablePrefixSum_;
    int32_t* HashTableHashBuckets_;

	int32_t* JoinTableHisto_;
	int32_t* JoinTablePrefixSum_;

	void* hash_prefix_sum_temp_buffer_;
	size_t hash_prefix_sum_temp_buffer_size_;

	void* join_prefix_sum_temp_buffer_;
	size_t join_prefix_sum_temp_buffer_size_;

public:
    GPUJoin(int32_t hashTableSize) :
		hashTableSize_(hashTableSize), 
		joinTableSize_(hashTableSize)
	{
        GPUMemory::alloc(&HashTableHisto_, hashTableSize_);
        GPUMemory::alloc(&HashTablePrefixSum_, hashTableSize_);
        GPUMemory::alloc(&HashTableHashBuckets_, hashTableSize_);

		GPUMemory::alloc(&JoinTableHisto_, joinTableSize_);
        GPUMemory::alloc(&JoinTablePrefixSum_, joinTableSize_);

		// Alloc the prefix sum helper buffers
		hash_prefix_sum_temp_buffer_ = nullptr;
		hash_prefix_sum_temp_buffer_size_ = 0;

		join_prefix_sum_temp_buffer_ = nullptr;
		join_prefix_sum_temp_buffer_size_ = 0;

		cub::DeviceScan::InclusiveSum(hash_prefix_sum_temp_buffer_, hash_prefix_sum_temp_buffer_size_, HashTableHisto_, HashTablePrefixSum_, hashTableSize_);
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&hash_prefix_sum_temp_buffer_), hash_prefix_sum_temp_buffer_size_);
	
		cub::DeviceScan::InclusiveSum(join_prefix_sum_temp_buffer_, join_prefix_sum_temp_buffer_size_, JoinTableHisto_, JoinTablePrefixSum_, joinTableSize_);
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&join_prefix_sum_temp_buffer_), join_prefix_sum_temp_buffer_size_);
	}

	~GPUJoin()
    {
        GPUMemory::free(HashTableHisto_);
        GPUMemory::free(HashTablePrefixSum_);
        GPUMemory::free(HashTableHashBuckets_);

		GPUMemory::free(JoinTableHisto_);
        GPUMemory::free(JoinTablePrefixSum_);

		GPUMemory::free(hash_prefix_sum_temp_buffer_);
		GPUMemory::free(join_prefix_sum_temp_buffer_);
    }

	template <typename T>
    void HashBlock(T* RTable, int32_t dataElementCount)
    {
		//////////////////////////////////////////////////////////////////////////////
		// Check for hash table limits
		if (dataElementCount < 0 || dataElementCount > hashTableSize_)
		{
            std::cerr << "Data element count exceeded hash table size" << std::endl;
            return;
		}

		//////////////////////////////////////////////////////////////////////////////
		// Calculate the hash table histograms
        kernel_calc_hash_histo<<<Context::getInstance().calcGridDim(hashTableSize_),
                                 Context::getInstance().getBlockDim()>>>(HashTableHisto_, 
																		 hashTableSize_,
                                                                         RTable, 
																		 dataElementCount);

		//////////////////////////////////////////////////////////////////////////////
        // Calculate the prefix sum for hashes

		/*
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;

        // Calculate the prefix sum
        cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, HashTableHisto_,
                                      HashTablePrefixSum_, hashTableSize_);
        
		// Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);
        
		// Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, HashTableHisto_,
                                      HashTablePrefixSum_, hashTableSize_);
        GPUMemory::free(tempBuffer);
		*/

		// Run inclusive prefix sum
		cub::DeviceScan::InclusiveSum(hash_prefix_sum_temp_buffer_, hash_prefix_sum_temp_buffer_size_, 
			HashTableHisto_, HashTablePrefixSum_, hashTableSize_);

		//////////////////////////////////////////////////////////////////////////////
        // Insert the keys into buckets
        kernel_put_data_to_buckets<<<Context::getInstance().calcGridDim(hashTableSize_),
                                     Context::getInstance().getBlockDim()>>>(HashTableHashBuckets_,
																			 HashTablePrefixSum_,
                                                                             hashTableSize_,
                                                                             RTable, 
																			 dataElementCount);
    }

	template<typename T>
    void JoinBlockCountMatches(int32_t* resultTableSize, T* RTable, int32_t dataElementCountRTable, T* STable, int32_t dataElementCountSTable)
	{
		//////////////////////////////////////////////////////////////////////////////
        // Check for join table limits
        if (dataElementCountSTable < 0 || dataElementCountSTable > joinTableSize_)
        {
            std::cerr << "Data element count exceeded join table size" << std::endl;
            return;
        }

        //////////////////////////////////////////////////////////////////////////////
        // Calculate the prbing result histograms
        kernel_calc_join_histo<<<Context::getInstance().calcGridDim(joinTableSize_),
                                 Context::getInstance().getBlockDim()>>>(JoinTableHisto_, 
																		 joinTableSize_,
                                                                         HashTableHisto_, 
																		 HashTablePrefixSum_,
                                                                         HashTableHashBuckets_, 
																		 hashTableSize_,
																		 RTable,
																		 dataElementCountRTable,
                                                                         STable, 
																		 dataElementCountSTable);

		//////////////////////////////////////////////////////////////////////////////
        // Calculate the prefix sum for probing results

		/*
        void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;

        // Calculate the prefix sum
        cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, JoinTableHisto_,
                                      JoinTablePrefixSum_, joinTableSize_);

        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

        // Run exclusive prefix sum
        cub::DeviceScan::InclusiveSum(tempBuffer, tempBufferSize, JoinTableHisto_,
                                      JoinTablePrefixSum_, joinTableSize_);
        GPUMemory::free(tempBuffer);
		*/

		cub::DeviceScan::InclusiveSum(join_prefix_sum_temp_buffer_, join_prefix_sum_temp_buffer_size_, 
			JoinTableHisto_, JoinTablePrefixSum_, joinTableSize_);
		
		//////////////////////////////////////////////////////////////////////////////
		// Calculate the result table size
		GPUMemory::copyDeviceToHost(resultTableSize, (JoinTablePrefixSum_ + joinTableSize_ - 1), 1);
	}

	template<typename T>
	void JoinBlockWriteResults(T* QTableA, T* QTableB, T* RTable, int32_t dataElementCountRTable, T* STable, int32_t dataElementCountSTable)
	{
		//////////////////////////////////////////////////////////////////////////////
		// Distribute the result data to the result buffer
		kernel_distribute_results_to_buffer << <Context::getInstance().calcGridDim(joinTableSize_),
			Context::getInstance().getBlockDim() >> > (QTableA,
													   QTableB,
													   JoinTableHisto_,
													   JoinTablePrefixSum_,
													   joinTableSize_,
													   HashTableHisto_,
													   HashTablePrefixSum_,
													   HashTableHashBuckets_,
													   hashTableSize_,
													   RTable,
													   dataElementCountRTable,
													   STable,
													   dataElementCountSTable);
	}
};