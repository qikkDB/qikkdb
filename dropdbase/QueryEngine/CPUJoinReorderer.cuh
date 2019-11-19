#pragma once

#include <cstdint>
#include <vector>
#include <iostream>
#include <thread>

#include "./GPUCore/GPUMemory.cuh"
#include "../ColumnBase.h"


class CPUJoinReorderer
{
public:
    template <typename T>
    static void reorderByJI(std::vector<T>& outBlock,
                            int32_t& outBlockSize,
                            const ColumnBase<T>& inCol,
                            int32_t inBlockIdx,
                            const std::vector<std::vector<int32_t>>& inColJoinIndices,
                            const int32_t blockSize)
    {
        outBlock.clear();
        outBlock.resize(inColJoinIndices[inBlockIdx].size());

        // Fetch the core count
        unsigned int threadCount = std::thread::hardware_concurrency();
        if (threadCount == 0)
        {
            throw std::runtime_error("[ERROR] Zero thread count returned by OS\n");
        }

        // Alloc the threads
        std::vector<std::thread> reorderThreads;
        for (int32_t idx = 0; idx < threadCount; idx++)
        {
            reorderThreads.push_back(
                std::thread{[](std::vector<T>& outBlock, const ColumnBase<T>& inCol, int32_t inBlockIdx,
                               const std::vector<std::vector<int32_t>>& inColJoinIndices,
                               const int32_t blockSize, const int32_t threadId, const int32_t threadCount) {
                                for (int32_t i = threadId; i < inColJoinIndices[inBlockIdx].size(); i += threadCount)
                                {
                                    const int32_t columnBlockId = inColJoinIndices[inBlockIdx][i] / blockSize;
                                    const int32_t columnRowId = inColJoinIndices[inBlockIdx][i] % blockSize;

                                    const T val = inCol.GetBlocksList()[columnBlockId]->GetData()[columnRowId];
                                    outBlock[i] = val;
                                }
                            },
                            std::ref(outBlock), std::ref(inCol), inBlockIdx,
                            std::ref(inColJoinIndices), blockSize, idx, threadCount});

		}

        for (auto& thread : reorderThreads)
        {
            thread.join();
        }

        outBlockSize = outBlock.size();
    }


    template <typename T>
    static void reorderByJIPushToGPU(T* outBlock,
                                     int32_t& outBlockSize,
                                     const ColumnBase<T>& inCol,
                                     int32_t inBlockIdx,
                                     const std::vector<std::vector<int32_t>>& inColJoinIndices,
                                     const int32_t blockSize)
    {

        std::vector<T> outBlockHost(inColJoinIndices[inBlockIdx].size());
        reorderByJI(outBlockHost, outBlockSize, inCol, inBlockIdx, inColJoinIndices, blockSize);

        GPUMemory::copyHostToDevice(outBlock, outBlockHost.data(), outBlockSize);
    }

    template <typename T>
    static void reorderNullMaskByJIPushToGPU(int8_t* outNullBlock,
                                             int32_t& outNullBlockSize,
                                             const ColumnBase<T>& inCol,
                                             int32_t inBlockIdx,
                                             const std::vector<std::vector<int32_t>>& inColJoinIndices,
                                             const int32_t blockSize)
    {
        // Alloc an output CPU vector
		outNullBlockSize = (inColJoinIndices[inBlockIdx].size() + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
        std::vector<int8_t> outNullBlockVector(outNullBlockSize);

		// Fetch the core count
        unsigned int threadCount = std::thread::hardware_concurrency();
        if (threadCount == 0)
        {
            throw std::runtime_error("[ERROR] Zero thread count returned by OS\n");
        }

        // Alloc the threads
        std::vector<std::thread> reorderNullThreads;
        for (int32_t idx = 0; idx < threadCount; idx++)
        {
            reorderNullThreads.push_back(
                std::thread{[](std::vector<int8_t>& outNullBlockVector, const ColumnBase<T>& inCol, int32_t inBlockIdx,
								const std::vector<std::vector<int32_t>>& inColJoinIndices,
								const int32_t blockSize, const int32_t threadId, const int32_t threadCount) {

								for (int32_t i = 8 * threadId; i < inColJoinIndices[inBlockIdx].size(); i += 8 * threadCount)
								{
									// Avodid write conflicts by assigning 8 rows to one core
									for(int32_t j = 0; j < 8; j++) {
										const int32_t columnBlockId = inColJoinIndices[inBlockIdx][i + j] / blockSize;
										const int32_t columnRowId = inColJoinIndices[inBlockIdx][i + j] % blockSize;

										const int8_t nullBit = (inCol.GetBlocksList()[columnBlockId]->GetNullBitmask()[columnRowId / (sizeof(int8_t) * 8)] >> (columnRowId % (sizeof(int8_t) * 8))) & 1;
                                
										outNullBlockVector[(i + j) / 8] |= (nullBit << j);
									}
								}
                            },
                            std::ref(outNullBlockVector), std::ref(inCol), inBlockIdx,
                            std::ref(inColJoinIndices), blockSize, idx, threadCount});

		}

        for (auto& thread : reorderNullThreads)
        {
            thread.join();
        }

		// Copy the result to the GPU
        GPUMemory::copyHostToDevice(outNullBlock, outNullBlockVector.data(), outNullBlockSize);
    }
};