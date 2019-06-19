#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <iostream>
#include <cstdint>
#include <vector>

#include "../dropdbase/ColumnBase.h"

TEST(GPUJoinTests, JoinTest)
{
	const int32_t SEED = 42;

	const int32_t PARSER_BLOCK_COUNT = 10;				// Number of blocks in the input
	const int32_t PARSER_BLOCK_SIZE = 1 << 10;			// The CUDA block size from the parser - simulated value

	ColumnBase<int32_t> ColumnR_("ColumnR", PARSER_BLOCK_SIZE);
	ColumnBase<int32_t> ColumnS_("ColumnS", PARSER_BLOCK_SIZE);

	// Fill the buffers with random data
	srand(SEED);

	for (int32_t i = 0; i < PARSER_BLOCK_COUNT; i++)
	{
		auto& blockR = ColumnR_.AddBlock();
		auto& blockS = ColumnS_.AddBlock();

		for (int32_t j = 0; j < PARSER_BLOCK_SIZE; j++)
		{
			blockR.InsertData(std::vector<int32_t>{rand()});
			blockS.InsertData(std::vector<int32_t>{rand()});
		}
	}

	// Run the join and store the result cross index in two vectors
	int32_t resultQTableSize;
	std::vector<int32_t> QATable;
	std::vector<int32_t> QBTable;

	GPUJoin::JoinTableRonS(QATable, QBTable, resultQTableSize, ColumnR_, ColumnS_, PARSER_BLOCK_SIZE);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	// DEBUG - convert tables to vectors
	std::vector<int32_t> RTable;
	std::vector<int32_t> STable;

	auto& ColumnRBlockList = ColumnR_.GetBlocksList();
	auto& ColumnSBlockList = ColumnS_.GetBlocksList();
	for (int32_t i = 0; i < PARSER_BLOCK_COUNT; i++)
	{
		auto& blockR = *ColumnRBlockList[i];
		auto& blockS = *ColumnSBlockList[i];
		for (int32_t j = 0; j < PARSER_BLOCK_SIZE; j++)
		{
			RTable.push_back(blockR.GetData()[j]);
			STable.push_back(blockS.GetData()[j]);
		}
	}
	
	std::printf("THIS: %d\n", ColumnS_.GetBlocksList()[0]->GetSize());
	for(int32_t i = 0; i < resultQTableSize; i++)
	{
		std::printf("%d %d\n", RTable[QATable[i]], STable[QBTable[i]]);
		ASSERT_EQ(RTable[QATable[i]], STable[QBTable[i]]);
	}
}

TEST(GPUJoinTests, ReorderCPUTest)
{
	/*
	// Run the join
	int32_t resultQTableSize;
	GPUJoin::JoinTableRonS(QATable, QBTable, resultQTableSize, RTable, STable);

	// Reorder - simulate the blockwise data input
	*/
}