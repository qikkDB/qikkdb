#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"

TEST(GPUJoinTests, JoinTest)
{
    // Initialize CUDA context:
    Context::getInstance();
   
	// Alloc buffers
    const int32_t blockCount = 2;		// Block count can be inf (GPU limits)
    const int32_t blockSize = 8;		// block size can be inf (GPU limits)
    int32_t fk[blockCount][blockSize] = {
		{5, 1, 0, 2, 0, 2, 1, 7},
		{0, 2, 5, 0, 1, 1, 7, 0}
	};

	// Alloc a feed buffer for building the hash table block - wise
    int32_t* d_feed_block;
    GPUMemory::alloc(&d_feed_block, blockSize);

	// Create a join instance to hold the hash table
	GPUJoin gpuJoin(blockCount, blockSize);

	// Copy the input block-wise and build the join hash table
    for (int32_t i = 0; i < blockCount; i++)
    {
		GPUMemory::copyHostToDevice(d_feed_block, fk[i], blockSize);
        gpuJoin.CalcBlockHashHisto(d_feed_block, i);
	}

	gpuJoin.CalcGlobalPrefixSum();

	/*
	for (int32_t i = 0; i < blockCount; i++)
    {
        GPUMemory::copyHostToDevice(d_feed_block, fk[i], blockSize);
        gpuJoin.SortHashTableIDs(d_feed_block, i);
    }
	*/

	gpuJoin.PrintDebug();

	GPUMemory::free(d_feed_block);

	ASSERT_EQ(false, true);
}