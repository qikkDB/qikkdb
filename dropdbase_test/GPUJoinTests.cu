#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

TEST(GPUJoinTests, JoinTest)
{
    const int32_t tableBlockCount = 1;
    const int32_t tableBlockSize = 16;
    const int32_t tableTotalSize = tableBlockSize * tableBlockCount;

    int32_t table[tableTotalSize] = {5, 1, 0, 2, 0, 2, 1, 7, 0, 2, 5, 0, 1, 1, 7, 0};

    int32_t* d_table;
    GPUMemory::alloc(&d_table, tableTotalSize);
    GPUMemory::copyHostToDevice(d_table, table, tableTotalSize);

    GPUJoin gpuJoin(tableBlockCount, tableBlockSize);
    for (int32_t i = 0; i < tableBlockCount; i++)
    {
        gpuJoin.HashBlock(&d_table[i * tableBlockSize], i);
	}

	gpuJoin.debugInfo();

    GPUMemory::free(d_table);

    ASSERT_EQ(false, true);
}