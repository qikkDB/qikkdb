#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUFilterConditions.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <iostream>
#include <cstdint>
#include <vector>

#include "../dropdbase/ColumnBase.h"

const int32_t SEED = 42;					// Random generator seed

const int32_t BLOCK_COUNT = 2;				// Number of blocks in the input
const int32_t BLOCK_SIZE = 1 << 11;			// The CUDA block size from the parser - simulated value

int32_t genVal(int32_t k, int32_t integerColumnCount)
{
	if (k % 2)
	{
		return k % (1024 * integerColumnCount);
	}
	else
	{
		return (k % (1024 * integerColumnCount)) * -1;
	}
}

TEST(GPUJoinTests, JoinTest)
{
	ColumnBase<int32_t> ColumnR_("ColumnR", BLOCK_SIZE);
	ColumnBase<int32_t> ColumnS_("ColumnS", BLOCK_SIZE);

	/*
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
	*/

	int32_t integerColumnCount_A = 1;
	int32_t integerColumnCount_B = 3;

	for (int32_t i = 0; i < BLOCK_COUNT; i++)
	{
		auto& blockR = ColumnR_.AddBlock();
		auto& blockS = ColumnS_.AddBlock();

		for (int32_t j = 0; j < BLOCK_SIZE; j++)
		{
			blockR.InsertData(std::vector<int32_t>{genVal(j, integerColumnCount_A)});
			blockS.InsertData(std::vector<int32_t>{genVal(j, integerColumnCount_B)});
		}
	}

	// Run the join and store the result cross index in two vectors of vectors each containing a vector of max BLOCK_SIZE
	std::vector<std::vector<int32_t>> resultColumnQAJoinIdx;
	std::vector<std::vector<int32_t>> resultColumnQBJoinIdx;

	bool aborted = false;
	GPUJoin::JoinTableRonS<FilterConditions::equal>(resultColumnQAJoinIdx, resultColumnQBJoinIdx, ColumnR_, ColumnS_, BLOCK_SIZE, aborted);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	for(int32_t i = 0; i < resultColumnQAJoinIdx.size(); i++)
	{
		for (int32_t j = 0; j < resultColumnQAJoinIdx[i].size(); j++)
		{
			int32_t RColumnBlockId = resultColumnQAJoinIdx[i][j] / BLOCK_SIZE;
			int32_t RColumnRowId = resultColumnQAJoinIdx[i][j] % BLOCK_SIZE;

			int32_t SColumnBlockId = resultColumnQBJoinIdx[i][j] / BLOCK_SIZE;
			int32_t SColumnRowId = resultColumnQBJoinIdx[i][j] % BLOCK_SIZE;

			int32_t val1 = ColumnR_.GetBlocksList()[RColumnBlockId]->GetData()[RColumnRowId];
			int32_t val2 = ColumnS_.GetBlocksList()[SColumnBlockId]->GetData()[SColumnRowId];

			// std:: cout << resultColumnQAJoinIdx[i][j] << " " << val1 << std::endl;
			// std:: cout << resultColumnQBJoinIdx[i][j] << " " << val2 << std::endl;
			// std:: cout << std::endl;

			ASSERT_EQ(val1, val2);
		}
	}
}

TEST(GPUJoinTests, JoinTestEmpty)
{
	ColumnBase<int32_t> ColumnR_("ColumnR", BLOCK_SIZE);
	ColumnBase<int32_t> ColumnS_("ColumnS", BLOCK_SIZE);

	for (int32_t i = 0; i < BLOCK_COUNT; i++)
	{
		auto& blockR = ColumnR_.AddBlock();
		auto& blockS = ColumnS_.AddBlock();

		for (int32_t j = 0; j < BLOCK_SIZE; j++)
		{
			blockR.InsertData(std::vector<int32_t>{0});
			blockS.InsertData(std::vector<int32_t>{1});
		}
	}

	// Run the join and store the result cross index in two vectors of vectors each containing a vector of max BLOCK_SIZE
	std::vector<std::vector<int32_t>> resultColumnQAJoinIdx;
	std::vector<std::vector<int32_t>> resultColumnQBJoinIdx;

	bool aborted = false;
	GPUJoin::JoinTableRonS<FilterConditions::equal>(resultColumnQAJoinIdx, resultColumnQBJoinIdx, ColumnR_, ColumnS_, BLOCK_SIZE, aborted);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 

	for(int32_t i = 0; i < resultColumnQAJoinIdx.size(); i++)
	{
		for (int32_t j = 0; j < resultColumnQAJoinIdx[i].size(); j++)
		{
			int32_t RColumnBlockId = resultColumnQAJoinIdx[i][j] / BLOCK_SIZE;
			int32_t RColumnRowId = resultColumnQAJoinIdx[i][j] % BLOCK_SIZE;

			int32_t SColumnBlockId = resultColumnQBJoinIdx[i][j] / BLOCK_SIZE;
			int32_t SColumnRowId = resultColumnQBJoinIdx[i][j] % BLOCK_SIZE;

			int32_t val1 = ColumnR_.GetBlocksList()[RColumnBlockId]->GetData()[RColumnRowId];
			int32_t val2 = ColumnS_.GetBlocksList()[SColumnBlockId]->GetData()[SColumnRowId];

			// std:: cout << resultColumnQAJoinIdx[i][j] << " " << val1 << std::endl;
			// std:: cout << resultColumnQBJoinIdx[i][j] << " " << val2 << std::endl;
			// std:: cout << std::endl;

			ASSERT_EQ(val1, val2);
		}
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

	// Run the join and store the result cross index in two vectors of vectors each containing a vector of max BLOCK_SIZE
	std::vector<std::vector<int32_t>> resultColumnQAJoinIdx;
	std::vector<std::vector<int32_t>> resultColumnQBJoinIdx;
	
	bool aborted = false;
	GPUJoin::JoinTableRonS<FilterConditions::equal>(resultColumnQAJoinIdx, resultColumnQBJoinIdx, ColumnR_, ColumnS_, BLOCK_SIZE, aborted);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Reorder the blocks
	int32_t *d_outRBlock;
	int32_t *d_outSBlock;

	GPUMemory::alloc(&d_outRBlock, BLOCK_SIZE);
	GPUMemory::alloc(&d_outSBlock, BLOCK_SIZE);

	int32_t outRBlockDataElementCount;
	int32_t outSBlockDataElementCount;

	std::vector<int32_t> outRBlock(BLOCK_SIZE);
	std::vector<int32_t> outSBlock(BLOCK_SIZE);

	int32_t outBlockCount = resultColumnQAJoinIdx.size();

	for (int32_t i = 0; i < outBlockCount; i++)
	{
		GPUJoin::reorderByJoinTableCPU(d_outRBlock, outRBlockDataElementCount, ColumnR_, i, resultColumnQAJoinIdx, BLOCK_SIZE);
		GPUJoin::reorderByJoinTableCPU(d_outSBlock, outSBlockDataElementCount, ColumnS_, i, resultColumnQBJoinIdx, BLOCK_SIZE);
	
		// Test if the results are equal
		GPUMemory::copyDeviceToHost(outRBlock.data(), d_outRBlock, outRBlockDataElementCount);
		GPUMemory::copyDeviceToHost(outSBlock.data(), d_outSBlock, outSBlockDataElementCount);
		for (int32_t j = 0; j < outRBlockDataElementCount; j++)
		{
			ASSERT_EQ(outRBlock[j], outSBlock[j]);
		}
	}
}

TEST(GPUJoinTests, ReorderEmptyCPUTest)
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
			blockR.InsertData(std::vector<int32_t>{0});
			blockS.InsertData(std::vector<int32_t>{1});
		}
	}

	// Run the join and store the result cross index in two vectors of vectors each containing a vector of max BLOCK_SIZE
	std::vector<std::vector<int32_t>> resultColumnQAJoinIdx;
	std::vector<std::vector<int32_t>> resultColumnQBJoinIdx;

    bool aborted = false;
	GPUJoin::JoinTableRonS<FilterConditions::equal>(resultColumnQAJoinIdx, resultColumnQBJoinIdx, ColumnR_, ColumnS_, BLOCK_SIZE, aborted);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Reorder the blocks
	int32_t *d_outRBlock;
	int32_t *d_outSBlock;

	GPUMemory::alloc(&d_outRBlock, BLOCK_SIZE);
	GPUMemory::alloc(&d_outSBlock, BLOCK_SIZE);

	int32_t outRBlockDataElementCount;
	int32_t outSBlockDataElementCount;

	std::vector<int32_t> outRBlock(BLOCK_SIZE);
	std::vector<int32_t> outSBlock(BLOCK_SIZE);

	int32_t outBlockCount = resultColumnQAJoinIdx.size();

	for (int32_t i = 0; i < outBlockCount; i++)
	{
		GPUJoin::reorderByJoinTableCPU(d_outRBlock, outRBlockDataElementCount, ColumnR_, i, resultColumnQAJoinIdx, BLOCK_SIZE);
		GPUJoin::reorderByJoinTableCPU(d_outSBlock, outSBlockDataElementCount, ColumnS_, i, resultColumnQBJoinIdx, BLOCK_SIZE);

		// Test if the results are equal
		GPUMemory::copyDeviceToHost(outRBlock.data(), d_outRBlock, outRBlockDataElementCount);
		GPUMemory::copyDeviceToHost(outSBlock.data(), d_outSBlock, outSBlockDataElementCount);
		for (int32_t j = 0; j < outRBlockDataElementCount; j++)
		{
			ASSERT_EQ(outRBlock[j], outSBlock[j]);
		}
	}
}