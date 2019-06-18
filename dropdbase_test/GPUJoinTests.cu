#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

TEST(GPUJoinTests, JoinTest)
{
	// Alloc the buffers
	const int32_t SEED = 42;

	// hash block size <= table size !!!!
	const size_t HASH_BLOCK_SIZE = 1 << 21;

	const size_t RTABLE_SIZE = 1 << 23;
	const size_t STABLE_SIZE = 1 << 23;

	std::vector<int32_t> RTable(RTABLE_SIZE);	// The first input table
	std::vector<int32_t> STable(STABLE_SIZE);	// The second input table

	std::vector<int32_t> QATable;				// The first result table
	std::vector<int32_t> QBTable;				// The second result table

	size_t QTableResultSizeTotal = 0;			// Total result size

	// Fill the buffers with random data
	srand(SEED);

	for (int32_t i = 0; i < RTABLE_SIZE; i++) { RTable[i] = rand(); }
	for (int32_t i = 0; i < STABLE_SIZE; i++) { STable[i] = rand(); }

	// Alloc GPU buffers
	int32_t *d_RTableBlock;
	int32_t *d_STableBlock;
	int32_t *d_QATableBlock;
	int32_t *d_QBTableBlock;

	GPUMemory::alloc(&d_RTableBlock, HASH_BLOCK_SIZE);
	GPUMemory::alloc(&d_STableBlock, HASH_BLOCK_SIZE);

	// Create a join instance
	GPUJoin gpuJoin(HASH_BLOCK_SIZE);

	// Perform the GPU join
	for (int32_t r = 0; r < RTABLE_SIZE; r += HASH_BLOCK_SIZE)
	{
		// For the last block process only the remaining elements
		int32_t processedRBlockSize = HASH_BLOCK_SIZE;
		if((RTABLE_SIZE - r) < HASH_BLOCK_SIZE)
		{
			processedRBlockSize = RTABLE_SIZE - r;
		}

		// Copy the first table block to the GPU and perform the hashing
		GPUMemory::copyHostToDevice(d_RTableBlock, &RTable[r], processedRBlockSize);
		gpuJoin.HashBlock(d_RTableBlock, processedRBlockSize);

		for (int32_t s = 0; s < STABLE_SIZE; s += HASH_BLOCK_SIZE)
		{
			// For the last block process only the remaining elements
			int32_t processedSBlockSize = HASH_BLOCK_SIZE;
			if((STABLE_SIZE - s) < HASH_BLOCK_SIZE)
			{
				processedSBlockSize = STABLE_SIZE - s;
			}

			// The result block size
			size_t processedQBlockResultSize = 0;

			// Copy the second table block to the GPU and perform the join
			// Calculate the required space
			GPUMemory::copyHostToDevice(d_STableBlock, &STable[s], processedSBlockSize);
			gpuJoin.JoinBlockCountMatches(&processedQBlockResultSize, d_RTableBlock, processedRBlockSize, d_STableBlock, processedSBlockSize);

			// Check if the result is not empty
			if(processedQBlockResultSize == 0)
			{
				continue;
			}

			// Alloc the result buffers
			GPUMemory::alloc(&d_QATableBlock, processedQBlockResultSize);
			GPUMemory::alloc(&d_QBTableBlock, processedQBlockResultSize);

			// Write the result data
			gpuJoin.JoinBlockWriteResults(d_QATableBlock, d_QBTableBlock, d_RTableBlock, processedRBlockSize, d_STableBlock, processedSBlockSize);

			// Copy the result blocks back and store them in the result set
			// The results can be at most n*n big
			int32_t *QAresult = new int32_t[processedQBlockResultSize];
			int32_t *QBresult = new int32_t[processedQBlockResultSize];

			GPUMemory::copyDeviceToHost(QAresult, d_QATableBlock, processedQBlockResultSize);
			GPUMemory::copyDeviceToHost(QBresult, d_QBTableBlock, processedQBlockResultSize);
			
			for(size_t i = 0; i < processedQBlockResultSize; i++)
			{
				QATable.push_back(r + QAresult[i]);	// Write the original idx
				QBTable.push_back(s + QBresult[i]);   // Write the original idx
				QTableResultSizeTotal++;
			}

			delete[] QAresult;
			delete[] QBresult;

			GPUMemory::free(d_QATableBlock);
			GPUMemory::free(d_QBTableBlock);

		}
	}

	GPUMemory::free(d_RTableBlock);
	GPUMemory::free(d_STableBlock);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	for(size_t i = 0; i < QTableResultSizeTotal; i++)
	{
		ASSERT_EQ(RTable[QATable[i]], STable[QBTable[i]]);
	}
}