#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <iostream>

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"
#include "GPUFilterConditions.cuh"

#include "../../ColumnBase.h"
#include "../../BlockBase.h"

#include "../../../cub/cub.cuh"


__device__ const int32_t HASH_TABLE_SUB_SIZE = 0x400;
__device__ const int32_t HASH_MOD = 0x3FF;

__device__ constexpr int32_t hash(int32_t key)
{
    return key & HASH_MOD;
}

template <typename T>
__global__ void
kernel_calc_hash_histo(int32_t* HashTableHisto, int32_t hashTableSize, T* ColumnRBlock, int8_t* nullBitMaskR, int32_t dataElementCount)
{
    __shared__ int32_t shared_memory[HASH_TABLE_SUB_SIZE];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < hashTableSize; i += stride)
    {
        // Count the occurances of hashes and accumulate them to the local memory
        shared_memory[threadIdx.x] = 0;
        __syncthreads();

        if (i < dataElementCount)
        {
            if (nullBitMaskR)
            {
                bool nullBitR = (nullBitMaskR[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                if (nullBitR)
                {
                    // Value in R col NULL - do nothing
                }
                else
                {
                    int32_t hash_idx = hash(ColumnRBlock[i]);
                    atomicAdd(&shared_memory[hash_idx], 1);
                }
            }
            else
            {
                int32_t hash_idx = hash(ColumnRBlock[i]);
                atomicAdd(&shared_memory[hash_idx], 1);
            }
        }

        __syncthreads();
        HashTableHisto[i] = shared_memory[threadIdx.x];
    }
}

template <typename T>
__global__ void kernel_put_data_to_buckets(int32_t* HashTableHashBuckets,
                                           int32_t* HashTablePrefixSum,
                                           int32_t hashTableSize,
                                           T* ColumnRBlock,
                                           int8_t* nullBitMaskR,
                                           int32_t dataElementCount)
{
    __shared__ int32_t shared_memory[HASH_TABLE_SUB_SIZE];

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < hashTableSize; i += stride)
    {
        // Set the prefix sum indexing counters into shared memory
        shared_memory[threadIdx.x] = (i == 0) ? 0 : HashTablePrefixSum[i - 1];
        __syncthreads();

        if (i < dataElementCount)
        {
            if (nullBitMaskR)
            {
                bool nullBitR = (nullBitMaskR[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                if (nullBitR)
                {
                    // Value in R col NULL - do nothing
                }
                else
                {
                    int32_t hash_idx = hash(ColumnRBlock[i]);
                    int32_t bucket_idx = atomicAdd(&shared_memory[hash_idx], 1);
                    HashTableHashBuckets[bucket_idx] = i; // ColumnRBlock[i];
                }
            }
            else
            {
                int32_t hash_idx = hash(ColumnRBlock[i]);
                int32_t bucket_idx = atomicAdd(&shared_memory[hash_idx], 1);
                HashTableHashBuckets[bucket_idx] = i; // ColumnRBlock[i];
            }
        }
    }
}

template <typename OP, typename T>
__global__ void kernel_calc_join_histo(int32_t* JoinTableHisto,
                                       int32_t joinTableSize,
                                       int32_t* HashTableHisto,
                                       int32_t* HashTablePrefixSum,
                                       int32_t* HashTableHashBuckets,
                                       int32_t hashTableSize,
                                       T* ColumnRBlock,
                                       int8_t* nullBitMaskR,
                                       int32_t dataElementCountColumnRBlock,
                                       T* ColumnSBlock,
                                       int8_t* nullBitMaskS,
                                       int32_t dataElementCountColumnSBlock)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < joinTableSize; i += stride)
    {
        // Zero the histo array
        // JoinTableHisto[i] = 0;

        // Count the number of result hash matches for this entry
        int32_t hashMatchCounter = 0;

        if (i < dataElementCountColumnSBlock)
        {
            // Hash table buckets probing and occurance counting
            int32_t hash_idx = hash(ColumnSBlock[i]);
            for (int32_t j = hash_idx; j < hashTableSize; j += HASH_TABLE_SUB_SIZE)
            {
                // Check if a bucket is empty, if yes, try the next bucket with the same hash
                // Otherwise probe and count the number of matching entries, mind the null values
                for (int32_t k = 0; k < HashTableHisto[j]; k++)
                {
                    if (nullBitMaskR)
                    {
                        bool nullBitR =
                            (nullBitMaskR[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                        if (nullBitR)
                        {
                            // Value in R col NULL - do nothing
                        }
                        else
                        {
                            if (nullBitMaskS)
                            {
                                bool nullBitS =
                                    (nullBitMaskS[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                                if (nullBitS)
                                {
                                    // Value in S col NULL - do nothing
                                }
                                else
                                {
                                    if (OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                             ColumnSBlock[i]))
                                    {
                                        hashMatchCounter++;
                                    }
                                }
                            }
                            else
                            {
                                if (OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                         ColumnSBlock[i]))
                                {
                                    hashMatchCounter++;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (nullBitMaskS)
                        {
                            bool nullBitS =
                                (nullBitMaskS[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                            if (nullBitS)
                            {
                                // Value in S col NULL - do nothing
                            }
                            else
                            {
                                if (OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                         ColumnSBlock[i]))
                                {
                                    hashMatchCounter++;
                                }
                            }
                        }
                        else
                        {
                            if (OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                     ColumnSBlock[i]))
                            {
                                hashMatchCounter++;
                            }
                        }
                    }
                }
            }
        }
        JoinTableHisto[i] = hashMatchCounter;
    }
}

template <typename OP, typename T>
__global__ void kernel_distribute_results_to_buffer(int32_t* resultColumnQABlockIdx,
                                                    int32_t* resultColumnQBBlockIdx,
                                                    int32_t* JoinTableHisto,
                                                    int32_t* JoinTablePrefixSum,
                                                    int32_t joinTableSize,
                                                    int32_t* HashTableHisto,
                                                    int32_t* HashTablePrefixSum,
                                                    int32_t* HashTableHashBuckets,
                                                    int32_t hashTableSize,
                                                    T* ColumnRBlock,
                                                    int8_t* nullBitMaskR,
                                                    int32_t dataElementCountColumnRBlock,
                                                    T* ColumnSBlock,
                                                    int8_t* nullBitMaskS,
                                                    int32_t dataElementCountColumnSBlock)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < joinTableSize; i += stride)
    {
        if (i < dataElementCountColumnSBlock)
        {
            int32_t join_prefix_sum_offset_index = 0;
            // Hash table buckets probing
            int32_t hash_idx = hash(ColumnSBlock[i]);
            for (int32_t j = hash_idx; j < hashTableSize; j += HASH_TABLE_SUB_SIZE)
            {
                // Check if a bucket is empty, if yes, try the next bucket with the same hash
                // Otherwise write count the number of matching entries
                for (int32_t k = 0; k < HashTableHisto[j]; k++)
                {
                    // Write the results if the value in a bucket matches the present value
                    // Write them to the calculated offset of the prefix sum buffer
                    if (nullBitMaskR)
                    {
                        bool nullBitR =
                            (nullBitMaskR[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                        if (nullBitR)
                        {
                            // Value in R col NULL - do nothing
                        }
                        else
                        {
                            if (nullBitMaskS)
                            {
                                bool nullBitS =
                                    (nullBitMaskS[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                                if (nullBitS)
                                {
                                    // Value in S col NULL - do nothing
                                }
                                else
                                {
                                    if (join_prefix_sum_offset_index < JoinTableHisto[i] &&
                                        OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                             ColumnSBlock[i]))
                                    {
                                        resultColumnQABlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                            HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k];
                                        resultColumnQBBlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                            i; // ColumnSBlock[i];
                                        join_prefix_sum_offset_index++;
                                    }
                                }
                            }
                            else
                            {
                                if (join_prefix_sum_offset_index < JoinTableHisto[i] &&
                                    OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                         ColumnSBlock[i]))
                                {
                                    resultColumnQABlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                        HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k];
                                    resultColumnQBBlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                        i; // ColumnSBlock[i];
                                    join_prefix_sum_offset_index++;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (nullBitMaskS)
                        {
                            bool nullBitS =
                                (nullBitMaskS[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
                            if (nullBitS)
                            {
                                // Value in S col NULL - do nothing
                            }
                            else
                            {
                                if (join_prefix_sum_offset_index < JoinTableHisto[i] &&
                                    OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                         ColumnSBlock[i]))
                                {
                                    resultColumnQABlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                        HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k];
                                    resultColumnQBBlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                        i; // ColumnSBlock[i];
                                    join_prefix_sum_offset_index++;
                                }
                            }
                        }
                        else
                        {
                            if (join_prefix_sum_offset_index < JoinTableHisto[i] &&
                                OP{}(ColumnRBlock[HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k]],
                                     ColumnSBlock[i]))
                            {
                                resultColumnQABlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                    HashTableHashBuckets[((j == 0) ? 0 : HashTablePrefixSum[j - 1]) + k];
                                resultColumnQBBlockIdx[((i == 0) ? 0 : JoinTablePrefixSum[i - 1]) + join_prefix_sum_offset_index] =
                                    i; // ColumnSBlock[i];
                                join_prefix_sum_offset_index++;
                            }
                        }
                    }
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

    template <typename T>
    void HashBlock(T* ColumnRBlock, int8_t* nullBitMaskR, int32_t dataElementCount)
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
        GPUMemory::memset(HashTableHisto_, 0, hashTableSize_);
        kernel_calc_hash_histo<<<Context::getInstance().calcGridDim(hashTableSize_),
                                 Context::getInstance().getBlockDim()>>>(HashTableHisto_, hashTableSize_,
                                                                         ColumnRBlock, nullBitMaskR,
                                                                         dataElementCount);

        //////////////////////////////////////////////////////////////////////////////
        // Calculate the prefix sum for hashes

        // Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum(hash_prefix_sum_temp_buffer_, hash_prefix_sum_temp_buffer_size_,
                                      HashTableHisto_, HashTablePrefixSum_, hashTableSize_);

        //////////////////////////////////////////////////////////////////////////////
        // Insert the keys into buckets
        kernel_put_data_to_buckets<<<Context::getInstance().calcGridDim(hashTableSize_),
                                     Context::getInstance().getBlockDim()>>>(HashTableHashBuckets_,
                                                                             HashTablePrefixSum_, hashTableSize_,
                                                                             ColumnRBlock, nullBitMaskR,
                                                                             dataElementCount);
    }

    template <typename OP, typename T>
    void JoinBlockCountMatches(int32_t* resultTableSize,
                               T* ColumnRBlock,
                               int8_t* nullBitMaskR,
                               int32_t dataElementCountColumnRBlock,
                               T* ColumnSBlock,
                               int8_t* nullBitMaskS,
                               int32_t dataElementCountColumnSBlock)
    {
        //////////////////////////////////////////////////////////////////////////////
        // Check for join table limits
        if (dataElementCountColumnSBlock < 0 || dataElementCountColumnSBlock > joinTableSize_)
        {
            std::cerr << "Data element count exceeded join table size" << std::endl;
            return;
        }

        //////////////////////////////////////////////////////////////////////////////
        // Calculate the prbing result histograms
        kernel_calc_join_histo<OP>
            <<<Context::getInstance().calcGridDim(joinTableSize_), Context::getInstance().getBlockDim()>>>(
                JoinTableHisto_, joinTableSize_, HashTableHisto_, HashTablePrefixSum_,
                HashTableHashBuckets_, hashTableSize_, ColumnRBlock, nullBitMaskR,
                dataElementCountColumnRBlock, ColumnSBlock, nullBitMaskS, dataElementCountColumnSBlock);

        //////////////////////////////////////////////////////////////////////////////
        // Calculate the prefix sum for probing results

        cub::DeviceScan::InclusiveSum(join_prefix_sum_temp_buffer_, join_prefix_sum_temp_buffer_size_,
                                      JoinTableHisto_, JoinTablePrefixSum_, joinTableSize_);

        //////////////////////////////////////////////////////////////////////////////
        // Calculate the result table size
        GPUMemory::copyDeviceToHost(resultTableSize, (JoinTablePrefixSum_ + joinTableSize_ - 1), 1);
    }

    template <typename OP, typename T>
    void JoinBlockWriteResults(int32_t* resultColumnQABlockIdx,
                               int32_t* resultColumnQBBlockIdx,
                               T* ColumnRBlock,
                               int8_t* nullBitMaskR,
                               int32_t dataElementCountColumnRBlock,
                               T* ColumnSBlock,
                               int8_t* nullBitMaskS,
                               int32_t dataElementCountColumnSBlock)
    {
        //////////////////////////////////////////////////////////////////////////////
        // Distribute the result data to the result buffer
        kernel_distribute_results_to_buffer<OP>
            <<<Context::getInstance().calcGridDim(joinTableSize_), Context::getInstance().getBlockDim()>>>(
                resultColumnQABlockIdx, resultColumnQBBlockIdx, JoinTableHisto_, JoinTablePrefixSum_,
                joinTableSize_, HashTableHisto_, HashTablePrefixSum_, HashTableHashBuckets_,
                hashTableSize_, ColumnRBlock, nullBitMaskR, dataElementCountColumnRBlock,
                ColumnSBlock, nullBitMaskS, dataElementCountColumnSBlock);
    }

public:
    GPUJoin(int32_t hashTableSize);

    ~GPUJoin();

    template <typename OP, typename T>
    static void JoinTableRonS(std::vector<std::vector<int32_t>>& resultColumnQAJoinIdx,
                              std::vector<std::vector<int32_t>>& resultColumnQBJoinIdx,
                              ColumnBase<T>& ColumnR,
                              ColumnBase<T>& ColumnS,
                              int32_t blockSize)
    {
        // The result vector - reset
        resultColumnQAJoinIdx.resize(0);
        resultColumnQBJoinIdx.resize(0);

        // Alloc the first block
        int32_t currentQAResultBlockIdx = 0;
        int32_t currentQBResultBlockIdx = 0;

        // Create a join instance
        GPUJoin gpuJoin(blockSize);

        // Alloc GPU input block buffers
        cuda_ptr<T> d_ColumnRBlock(blockSize);
        cuda_ptr<T> d_ColumnSBlock(blockSize);

        size_t nullColSizeRBlock = (blockSize + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
        size_t nullColSizeSBlock = (blockSize + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);

        cuda_ptr<int8_t> d_ColumnNullRBlock(nullColSizeRBlock);
        cuda_ptr<int8_t> d_ColumnNullSBlock(nullColSizeSBlock);

        // Perform the GPU join
        auto& ColumnRBlockList = ColumnR.GetBlocksList();
        auto& ColumnSBlockList = ColumnS.GetBlocksList();
        for (int32_t r = 0; r < ColumnR.GetBlockCount(); r++)
        {
            // For the last block process only the remaining elements
            int32_t processedRBlockSize = ColumnRBlockList[r]->GetSize();
            int32_t processedNullRBlockSize = ColumnRBlockList[r]->GetNullBitmaskSize();

            // Copy the first table block to the GPU and perform the hashing
            GPUMemory::copyHostToDevice(d_ColumnRBlock.get(), ColumnRBlockList[r]->GetData(), processedRBlockSize);

            // Configure arguments based on null values
            if (ColumnRBlockList[r]->GetNullBitmask())
            {
                GPUMemory::copyHostToDevice(d_ColumnNullRBlock.get(),
                                            ColumnRBlockList[r]->GetNullBitmask(), processedNullRBlockSize);
                gpuJoin.HashBlock(d_ColumnRBlock.get(), d_ColumnNullRBlock.get(), processedRBlockSize);
            }
            else
            {
                gpuJoin.HashBlock(d_ColumnRBlock.get(), nullptr, processedRBlockSize);
            }

            GPUMemory::memset(gpuJoin.JoinTableHisto_, 0, gpuJoin.joinTableSize_);
            for (int32_t s = 0; s < ColumnS.GetBlockCount(); s++)
            {
                // The result block size
                int32_t processedQBlockResultSize = 0;

                // For the last block process only the remaining elements
                int32_t processedSBlockSize = ColumnSBlockList[s]->GetSize();
                int32_t processedNullSBlockSize = ColumnSBlockList[s]->GetNullBitmaskSize();

                // Copy the second table block to the GPU and perform the join
                // Calculate the required space
                GPUMemory::copyHostToDevice(d_ColumnSBlock.get(), ColumnSBlockList[s]->GetData(), processedSBlockSize);

                // Configure arguments based on null values
                if (ColumnSBlockList[s]->GetNullBitmask())
                {
                    GPUMemory::copyHostToDevice(d_ColumnNullSBlock.get(),
                                                ColumnSBlockList[s]->GetNullBitmask(), processedNullSBlockSize);

                    if (ColumnRBlockList[r]->GetNullBitmask())
                    {
                        gpuJoin.JoinBlockCountMatches<OP>(&processedQBlockResultSize,
                                                          d_ColumnRBlock.get(), d_ColumnNullRBlock.get(),
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          d_ColumnNullSBlock.get(), processedSBlockSize);
                    }
                    else
                    {
                        gpuJoin.JoinBlockCountMatches<OP>(&processedQBlockResultSize,
                                                          d_ColumnRBlock.get(), nullptr,
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          d_ColumnNullSBlock.get(), processedSBlockSize);
                    }
                }
                else
                {
                    if (ColumnRBlockList[r]->GetNullBitmask())
                    {
                        gpuJoin.JoinBlockCountMatches<OP>(&processedQBlockResultSize,
                                                          d_ColumnRBlock.get(), d_ColumnNullRBlock.get(),
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          nullptr, processedSBlockSize);
                    }
                    else
                    {
                        gpuJoin.JoinBlockCountMatches<OP>(&processedQBlockResultSize,
                                                          d_ColumnRBlock.get(), nullptr,
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          nullptr, processedSBlockSize);
                    }
                }

                // Check if the result is not empty
                if (processedQBlockResultSize == 0)
                {
                    continue;
                }

                // Alloc the result buffers
                cuda_ptr<int32_t> d_QAResultBlock(processedQBlockResultSize);
                cuda_ptr<int32_t> d_QBResultBlock(processedQBlockResultSize);

                // Write the result data - configure args according to null vals
                if (ColumnSBlockList[s]->GetNullBitmask())
                {
                    if (ColumnRBlockList[r]->GetNullBitmask())
                    {
                        gpuJoin.JoinBlockWriteResults<OP>(d_QAResultBlock.get(), d_QBResultBlock.get(),
                                                          d_ColumnRBlock.get(), d_ColumnNullRBlock.get(),
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          d_ColumnNullSBlock.get(), processedSBlockSize);
                    }
                    else
                    {
                        gpuJoin.JoinBlockWriteResults<OP>(d_QAResultBlock.get(), d_QBResultBlock.get(),
                                                          d_ColumnRBlock.get(), nullptr,
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          d_ColumnNullSBlock.get(), processedSBlockSize);
                    }
                }
                else
                {
                    if (ColumnRBlockList[r]->GetNullBitmask())
                    {
                        gpuJoin.JoinBlockWriteResults<OP>(d_QAResultBlock.get(), d_QBResultBlock.get(),
                                                          d_ColumnRBlock.get(), d_ColumnNullRBlock.get(),
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          nullptr, processedSBlockSize);
                    }
                    else
                    {
                        gpuJoin.JoinBlockWriteResults<OP>(d_QAResultBlock.get(), d_QBResultBlock.get(),
                                                          d_ColumnRBlock.get(), nullptr,
                                                          processedRBlockSize, d_ColumnSBlock.get(),
                                                          nullptr, processedSBlockSize);
                    }
                }

                // Copy the result blocks back and store them in the result set
                // The results can be at most n*n big
                std::vector<int32_t> QAresult(processedQBlockResultSize);
                std::vector<int32_t> QBresult(processedQBlockResultSize);

                GPUMemory::copyDeviceToHost(&QAresult[0], d_QAResultBlock.get(), processedQBlockResultSize);
                GPUMemory::copyDeviceToHost(&QBresult[0], d_QBResultBlock.get(), processedQBlockResultSize);

                for (int32_t i = 0; i < processedQBlockResultSize; i++)
                {
                    if (resultColumnQAJoinIdx.size() == 0)
                    {
                        resultColumnQAJoinIdx.push_back(std::vector<int32_t>{});
                    }

                    if (resultColumnQAJoinIdx[currentQAResultBlockIdx].size() == blockSize)
                    {
                        resultColumnQAJoinIdx.push_back(std::vector<int32_t>{});
                        currentQAResultBlockIdx++;
                    }

                    if (resultColumnQBJoinIdx.size() == 0)
                    {
                        resultColumnQBJoinIdx.push_back(std::vector<int32_t>{});
                    }

                    if (resultColumnQBJoinIdx[currentQBResultBlockIdx].size() == blockSize)
                    {
                        resultColumnQBJoinIdx.push_back(std::vector<int32_t>{});
                        currentQBResultBlockIdx++;
                    }

                    resultColumnQAJoinIdx[currentQAResultBlockIdx].push_back(r * blockSize + QAresult[i]); // Write the original idx
                    resultColumnQBJoinIdx[currentQBResultBlockIdx].push_back(s * blockSize + QBresult[i]); // Write the original idx
                }
            }
        }
    }

    // Create a new outBlock based on a portion of join indexes and input column
    template <typename T>
    static void reorderByJoinTableCPU(T* outBlock,
                                      int32_t& outDataSize,
                                      const ColumnBase<T>& inColumn,
                                      int32_t resultColumnQJoinIdxBlockIdx,
                                      const std::vector<std::vector<int32_t>>& resultColumnQJoinIdx,
                                      int32_t blockSize)
    {
        if (resultColumnQJoinIdxBlockIdx < 0 || resultColumnQJoinIdxBlockIdx > resultColumnQJoinIdx.size())
        {
            std::cerr << "[ERROR]  Column block index out of bounds" << std::endl;
        }

        // Allocan output CPU vector
        std::vector<T> outBlockVector(resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx].size());

        for (int32_t i = 0; i < resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx].size(); i++)
        {
            int32_t columnBlockId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] / blockSize;
            int32_t columnRowId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] % blockSize;

            T val = inColumn.GetBlocksList()[columnBlockId]->GetData()[columnRowId];
            outBlockVector[i] = val;
        }

        outDataSize = outBlockVector.size();

        GPUMemory::copyHostToDevice(outBlock, outBlockVector.data(), outDataSize);
    }

    // Create a new outBlock based on a portion of join indexes and input column
    template <typename T>
    static void reorderByJoinTableCPU(std::vector<T>& outBlock,
                                      int32_t& outDataSize,
                                      const ColumnBase<T>& inColumn,
                                      int32_t resultColumnQJoinIdxBlockIdx,
                                      const std::vector<std::vector<int32_t>>& resultColumnQJoinIdx,
                                      int32_t blockSize)
    {
        if (resultColumnQJoinIdxBlockIdx < 0 || resultColumnQJoinIdxBlockIdx > resultColumnQJoinIdx.size())
        {
            std::cerr << "[ERROR]  Column block index out of bounds" << std::endl;
        }

        outBlock.clear();

        for (int32_t i = 0; i < resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx].size(); i++)
        {
            int32_t columnBlockId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] / blockSize;
            int32_t columnRowId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] % blockSize;

            T val = inColumn.GetBlocksList()[columnBlockId]->GetData()[columnRowId];
            outBlock.push_back(val);
        }
        outDataSize = outBlock.size();
    }

    // Reorder the null mask and put it in a GPU buffer
    template <typename T>
    static void reorderNullMaskByJoinTableCPU(int8_t* outNullBlock,
                                              int32_t& outNullBlockSize,
                                              const ColumnBase<T>& inColumn,
                                              int32_t resultColumnQJoinIdxBlockIdx,
                                              const std::vector<std::vector<int32_t>>& resultColumnQJoinIdx,
                                              int32_t blockSize)
    {
        if (resultColumnQJoinIdxBlockIdx < 0 || resultColumnQJoinIdxBlockIdx > resultColumnQJoinIdx.size())
        {
            std::cerr << "[ERROR]  Column block index out of bounds" << std::endl;
        }

        // Allocan output CPU vector
        std::vector<int8_t> outNullBlockVector(
            (resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx].size() + sizeof(int8_t) * 8 - 1) /
            (sizeof(int8_t) * 8));

        for (int32_t i = 0; i < resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx].size(); i++)
        {
            int32_t columnBlockId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] / blockSize;
            int32_t columnRowId = resultColumnQJoinIdx[resultColumnQJoinIdxBlockIdx][i] % blockSize;

            int8_t nullBit =
                (inColumn.GetBlocksList()[columnBlockId]->GetNullBitmask()[columnRowId / (sizeof(int8_t) * 8)] >>
                 (columnRowId % (sizeof(int8_t) * 8))) &
                1;

            nullBit <<= (i % (sizeof(int8_t) * 8));
            outNullBlockVector[i / 8] |= nullBit;
        }

        outNullBlockSize = outNullBlockVector.size();

        GPUMemory::copyHostToDevice(outNullBlock, outNullBlockVector.data(), outNullBlockSize);
    }
};