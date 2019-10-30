#pragma once

#include <cstdint>
#include <vector>
#include <iostream>


#include "./GPUCore/GPUMemory.cuh"
#include "../ColumnBase.h"


class CPUJoinReorderer {
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
        
        for (int32_t i = 0; i < inColJoinIndices[inBlockIdx].size(); i++)
        {
            int32_t columnBlockId = inColJoinIndices[inBlockIdx][i] / blockSize;
            int32_t columnRowId = inColJoinIndices[inBlockIdx][i] % blockSize;

            T val = inCol.GetBlocksList()[columnBlockId]->GetData()[columnRowId];
            outBlock[i] = val;
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
											 const ColumnBase<T>& inColumn,
											 int32_t resultColumnQJoinIdxBlockIdx,
                                             const std::vector<std::vector<int32_t>>& resultColumnQJoinIdx,
                                             const int32_t blockSize)
    {
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