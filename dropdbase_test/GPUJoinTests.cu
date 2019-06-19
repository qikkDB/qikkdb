#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

class GPUJoinTests : public ::testing::Test
{
public:
	// Alloc the buffers
	const int32_t SEED = 42;

	const int32_t PARSER_BLOCK_SIZE = 1 << 30;		// The CUDA block size from the parser - simulated value
	const int32_t HASH_BLOCK_SIZE = 1 << 20;	// THe hash table to itterazte trough the input data in a O(n^2) cycle

	const int32_t RTABLE_SIZE = 1 << 20;		// The first block must be the BIGGER ONE !!!!
	const int32_t STABLE_SIZE = 1 << 10;

	std::vector<int32_t> RTable;	// The first input table
	std::vector<int32_t> STable;	// The second input table

	std::vector<int32_t> QATable;	// The first result table
	std::vector<int32_t> QBTable;	// The second result table

	virtual void SetUp()
	{
		// Fill the buffers with random data
		srand(SEED);

		for (int32_t i = 0; i < RTABLE_SIZE; i++) { RTable.push_back(rand()); }
		for (int32_t i = 0; i < STABLE_SIZE; i++) { STable.push_back(rand()); }
	}

	virtual void TearDown()
	{


	}
};

TEST_F(GPUJoinTests, JoinTest)
{
	// Run the join
	int32_t resultQTableSize;
	GPUJoin::JoinTableRonS(QATable, QBTable, resultQTableSize, RTable, STable);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Check the results 
	
	for(int32_t i = 0; i < resultQTableSize; i++)
	{
		ASSERT_EQ(RTable[QATable[i]], STable[QBTable[i]]);
	}
	
}

TEST_F(GPUJoinTests, ReorderCPUTest)
{
	// Run the join
	int32_t resultQTableSize;
	GPUJoin::JoinTableRonS(QATable, QBTable, resultQTableSize, RTable, STable);

	// Reorder - simulate the blockwise data input
}