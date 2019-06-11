#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

TEST(GPUJoinTests, JoinTest)
{
    const int32_t resultTablePageSize = 2;

    const int32_t tableABlockCount = 1;
    const int32_t tableABlockSize = 16;
    const int32_t tableATotalSize = tableABlockSize * tableABlockCount;

	const int32_t tableBBlockCount = 1;
    const int32_t tableBBlockSize = 8;
    const int32_t tableBTotalSize = tableBBlockSize * tableBBlockCount;

    int32_t tableA[tableATotalSize] = {5, 1, 0, 2, 0, 2, 1, 7, 0, 2, 5, 0, 1, 1, 7, 0};
    int32_t tableB[tableBTotalSize] = {5, 1, 7, 0, 2, 3, 4, 6};

    int32_t* d_tableA;
    int32_t* d_tableB;

    GPUMemory::alloc(&d_tableA, tableATotalSize);
    GPUMemory::alloc(&d_tableB, tableBTotalSize);

    GPUMemory::copyHostToDevice(d_tableA, tableA, tableATotalSize);
    GPUMemory::copyHostToDevice(d_tableB, tableB, tableBTotalSize);

	// Create a join instance
    GPUJoin gpuJoin(tableABlockCount, tableABlockSize, resultTablePageSize);

	// Build hash table
    for (int32_t i = 0; i < tableABlockCount; i++)
    {
        gpuJoin.HashBlock(&d_tableA[i * tableABlockSize], i);
	}

	// Join the tables
    for (int32_t i = 0; i < tableBBlockCount; i++)
    {
        gpuJoin.JoinBlockOnHashTable(&tableB[i * tableBBlockSize], tableBBlockSize);
    }

	gpuJoin.debugInfo();

    GPUMemory::free(d_tableA);

    ASSERT_EQ(false, true);
}