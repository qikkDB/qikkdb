#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

void basicUsage() 
{
	// DO NOT DELETE
	const int32_t blockSize = 8;
	const int32_t hashTableSize = blockSize;

	const int32_t RTableDataElementCount = 16;
	const int32_t STableDataElementCount = 8;
	const int32_t QTableDataElementCount = blockSize;

	int32_t RTable[RTableDataElementCount] = { 5, 1, 0, 2, 0, 2, 1, 7, 0, 2, 5, 0, 1, 1, 7, 0 };
	int32_t STable[STableDataElementCount] = { 5, 5, 5, 5, 5, 5, 5, 5 };//{ 5, 1, 7, 0, 2, 3, 4, 6 };

	int32_t QATable[QTableDataElementCount];
	int32_t QBTable[QTableDataElementCount];

	int32_t* d_RTable;
	int32_t* d_STable;
	int32_t* d_QATable;
	int32_t* d_QBTable;

	GPUMemory::alloc(&d_RTable, RTableDataElementCount);
	GPUMemory::alloc(&d_STable, STableDataElementCount);
	GPUMemory::alloc(&d_QATable, QTableDataElementCount);
	GPUMemory::alloc(&d_QBTable, QTableDataElementCount);

	GPUMemory::copyHostToDevice(d_RTable, RTable, RTableDataElementCount);
	GPUMemory::copyHostToDevice(d_STable, STable, STableDataElementCount);

	// Create a join instance
	GPUJoin gpuJoin(hashTableSize);

	// Hash the values and then join them
	for (int32_t i = 0; i < RTableDataElementCount; i += blockSize)
	{
		gpuJoin.HashBlock(&d_RTable[i], blockSize);
		for (int32_t j = 0; j < STableDataElementCount; j += blockSize)
		{
			int32_t resultSize;
			gpuJoin.JoinBlock(d_QATable, d_QBTable, &resultSize, &d_RTable[i], blockSize, &d_STable[j], blockSize);

			// DEBUG - Copy the blocks back and print their content
			GPUMemory::copyDeviceToHost(QATable, d_QATable, QTableDataElementCount);
			GPUMemory::copyDeviceToHost(QBTable, d_QBTable, QTableDataElementCount);

			std::printf("#### RESULT INFO ###\n");
			for (int32_t debug_idx = 0; debug_idx < QTableDataElementCount; debug_idx++)
			{
				std::printf("%d %d\n", QATable[debug_idx], QBTable[debug_idx]);
			}
		}

		// DEBUG - Info about inner buffers
		gpuJoin.printDebugInfo();
	}

	GPUMemory::free(d_RTable);
	GPUMemory::free(d_STable);
	GPUMemory::free(d_QATable);
	GPUMemory::free(d_QBTable);
}


TEST(GPUJoinTests, JoinTest)
{

	// Alloc the buffers
	const int32_t SEED = 42;

	// hash block size <= table size !!!!
	const int32_t HASH_BLOCK_SIZE = 1 << 12;

	const int32_t RTABLE_SIZE = 1 << 20;
	const int32_t STABLE_SIZE = 1 << 20;
	const int32_t QTABLE_SIZE = std::max(RTABLE_SIZE, STABLE_SIZE);

	std::vector<int32_t> RTable(RTABLE_SIZE);	// The first input table
	std::vector<int32_t> STable(STABLE_SIZE);	// The second input table

	std::vector<int32_t> QATable(QTABLE_SIZE);	// Result table A - first table join indexes
	std::vector<int32_t> QBTable(QTABLE_SIZE);  // Result table B - second table join indexes

	int32_t QTableResultSizeTotal = 0;			// Total result size

	// Fill the buffers with random data
	srand(SEED);

	for (int32_t i = 0; i < RTABLE_SIZE; i++) { RTable[i] = rand(); }
	for (int32_t i = 0; i < STABLE_SIZE; i++) { STable[i] = rand(); }
	for (int32_t i = 0; i < QTABLE_SIZE; i++) { QATable[i] = 0; QBTable[i] = 0; }

	// Alloc GPU buffers
	int32_t *d_RTableBlock;
	int32_t *d_STableBlock;
	int32_t *d_QATableBlock;
	int32_t *d_QBTableBlock;

	GPUMemory::alloc(&d_RTableBlock, HASH_BLOCK_SIZE);
	GPUMemory::alloc(&d_STableBlock, HASH_BLOCK_SIZE);
	GPUMemory::alloc(&d_QATableBlock, HASH_BLOCK_SIZE);
	GPUMemory::alloc(&d_QBTableBlock, HASH_BLOCK_SIZE);

	// Create a join instance
	GPUJoin gpuJoin(HASH_BLOCK_SIZE);

	// Perform the GPU join
	for (int32_t r = 0; r < RTABLE_SIZE; r += HASH_BLOCK_SIZE)
	{
		// For the last block process only the remaining elements
		int32_t processedRBlockSize = HASH_BLOCK_SIZE;
		if((r + HASH_BLOCK_SIZE) > RTABLE_SIZE)
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
			if((s + HASH_BLOCK_SIZE) > STABLE_SIZE)
			{
				processedSBlockSize = STABLE_SIZE - s;
			}

			// The result block size
			int32_t processedQBlockResultSize = 0;

			// Copy the second table block to the GPU and perform the join
			GPUMemory::copyHostToDevice(d_STableBlock, &STable[s], processedSBlockSize);
			gpuJoin.JoinBlock(d_QATableBlock, d_QBTableBlock, &processedQBlockResultSize, d_RTableBlock, processedRBlockSize, d_STableBlock, processedSBlockSize);

			// Copy the result blocks back and store them in the result set
			int32_t QAresult[HASH_BLOCK_SIZE];
			int32_t QBresult[HASH_BLOCK_SIZE];

			GPUMemory::copyDeviceToHost(QAresult, d_QATableBlock, processedQBlockResultSize);
			GPUMemory::copyDeviceToHost(QBresult, d_QBTableBlock, processedQBlockResultSize);
			
			for(int32_t i = 0; i < processedQBlockResultSize; i++)
			{
				QATable[QTableResultSizeTotal] = QAresult[i];
				QBTable[QTableResultSizeTotal] = QBresult[i];
				QTableResultSizeTotal++;
			}
		}
	}

	GPUMemory::free(d_RTableBlock);
	GPUMemory::free(d_STableBlock);
	GPUMemory::free(d_QATableBlock);
	GPUMemory::free(d_QBTableBlock);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	for(int32_t i = 0; i < QTableResultSizeTotal; i++)
	{
		std::printf("%d %d\n", QATable[i], QBTable[i]);

		if(i > 100) 
			break;
	}

	//basicUsage();
    FAIL();
}