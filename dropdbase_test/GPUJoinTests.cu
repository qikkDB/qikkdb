#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <iostream>
#include <cstdint>
#include <vector>

#include "../dropdbase/ColumnBase.h"

const int32_t SEED = 42;					// Random generator seed

const int32_t BLOCK_COUNT = 10;				// Number of blocks in the input
const int32_t BLOCK_SIZE = 1 << 10;			// The CUDA block size from the parser - simulated value

TEST(GPUJoinTests, JoinTest)
{
	ColumnBase<int32_t> ColumnR_("ColumnR", BLOCK_SIZE);
	ColumnBase<int32_t> ColumnS_("ColumnS", BLOCK_SIZE);

	// Fill the buffers with random data
	srand(SEED);

	for (int32_t i = 0; i < BLOCK_COUNT; i++)
	{
		auto& blockR = ColumnR_.AddBlock();
		auto& blockS = ColumnS_.AddBlock();

		for (int32_t j = 0; j < BLOCK_SIZE; j++)
		{
			blockR.InsertData(std::vector<int32_t>{rand()});
			blockS.InsertData(std::vector<int32_t>{rand()});
		}
	}

	// Run the join and store the result cross index in two vectors
	int32_t resultQTableSize;
	std::vector<int32_t> resultColumnQAJoinIdx;
	std::vector<int32_t> resultColumnQBJoinIdx;

	GPUJoin::JoinTableRonS(resultColumnQAJoinIdx, resultColumnQBJoinIdx, resultQTableSize, ColumnR_, ColumnS_, BLOCK_SIZE);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	// DEBUG - convert tables to vectors
	std::vector<int32_t> RTable;
	std::vector<int32_t> STable;

	auto& ColumnRBlockList = ColumnR_.GetBlocksList();
	auto& ColumnSBlockList = ColumnS_.GetBlocksList();
	for (int32_t i = 0; i < BLOCK_COUNT; i++)
	{
		auto& blockR = *ColumnRBlockList[i];
		auto& blockS = *ColumnSBlockList[i];
		for (int32_t j = 0; j < BLOCK_SIZE; j++)
		{
			RTable.push_back(blockR.GetData()[j]);
			STable.push_back(blockS.GetData()[j]);
		}
	}

	// Check the results
	for(int32_t i = 0; i < resultQTableSize; i++)
	{
		ASSERT_EQ(RTable[resultColumnQAJoinIdx[i]], STable[resultColumnQBJoinIdx[i]]);
	}
}

TEST(GPUJoinTests, ReorderCPUTest)
{
	ColumnBase<int32_t> ColumnR_("ColumnR", BLOCK_SIZE);
	ColumnBase<int32_t> ColumnS_("ColumnS", BLOCK_SIZE);

	// Fill the buffers with random data
	srand(SEED);

	for (int32_t i = 0; i < BLOCK_COUNT; i++)
	{
		auto& blockR = ColumnR_.AddBlock();
		auto& blockS = ColumnS_.AddBlock();

		for (int32_t j = 0; j < BLOCK_SIZE; j++)
		{
			blockR.InsertData(std::vector<int32_t>{rand()});
			blockS.InsertData(std::vector<int32_t>{rand()});
		}
	}

	// Run the join and store the result cross index in two vectors
	int32_t resultQTableSize;
	std::vector<int32_t> resultColumnQAJoinIdx;
	std::vector<int32_t> resultColumnQBJoinIdx;

	GPUJoin::JoinTableRonS(resultColumnQAJoinIdx, resultColumnQBJoinIdx, resultQTableSize, ColumnR_, ColumnS_, BLOCK_SIZE);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Reorder the columns - put the data from a vector into a blockbase
	for (int32_t i = 0; i < resultQTableSize; i += BLOCK_SIZE)
	{
		// Alloc and fill vectors with a subset of the result data
		std::vector<int32_t> resultColumnQAJoinIdxBlock;
		std::vector<int32_t> resultColumnQBJoinIdxBlock;

		for (int32_t j = 0; j < BLOCK_SIZE && (j + i) < resultQTableSize; j++)
		{
			resultColumnQAJoinIdxBlock.push_back(resultColumnQAJoinIdx[i + j]);
			resultColumnQBJoinIdxBlock.push_back(resultColumnQBJoinIdx[i + j]);
		}

		// Reconstruct the result based on the indexes


	}
}