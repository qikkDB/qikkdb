#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUOrderBy.cuh"

TEST(GPUOrderByTests, GPUOrderByTest)
{
    // Input data
    constexpr int32_t GROUP_BY_BLOCK_COUNT = 2;
    constexpr int32_t DATA_ELEMENT_COUNT = 16;
    int32_t inCols[GROUP_BY_BLOCK_COUNT][DATA_ELEMENT_COUNT] = {
        {34, 12, 34, 2, 78, 3, 34, 89, 78, 59, 12, 78, 90, 33, 78, 90}, 
        {21, 11, 49, 25, 8, 32, 71, 82, 0, 29, 11, 64, 4, 67, 22, 91}
    };

    // Alloc the input blocks and move the data
    std::vector<int32_t*> d_inCols(GROUP_BY_BLOCK_COUNT);
    for(int i = 0; i < GROUP_BY_BLOCK_COUNT; i++)
    {
        GPUMemory::alloc(&inCols[i], DATA_ELEMENT_COUNT);
        GPUMemory::copyHostToDevice(d_inCols[i], inCols[i], DATA_ELEMENT_COUNT);
    }

    // Result order by indices
    int32_t* outColIndices;
    GPUMemory::alloc(&outColIndices, DATA_ELEMENT_COUNT);

    //////////////////////////////////////////////////////////////////////////////
    // Run the order by operation

    GPUOrderBy orderBy;
    orderBy.OrderBy(outColIndices, d_inCols, DATA_ELEMENT_COUNT);

    //////////////////////////////////////////////////////////////////////////////
    // Dealloc the input blocks
    for(int i = 0; i < GROUP_BY_BLOCK_COUNT; i++)
    {
        GPUMemory::free(d_inCols[i]);
    }
    GPUMemory::free(outColIndices);
}
