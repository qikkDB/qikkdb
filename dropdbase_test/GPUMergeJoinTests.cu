#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
	// Initialize test buffers
	const int32_t BLOCK_COUNT_A = 16;
	const int32_t BLOCK_SIZE_A = 1 << 25;	

	const int32_t BLOCK_COUNT_B = 16;
	const int32_t BLOCK_SIZE_B = 1 << 25;	

	ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A);
	ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B);

	for (int32_t i = 0; i < BLOCK_COUNT_A; i++)
	{
		auto& blockA = colA.AddBlock();

		std::vector<int32_t> colAData;
		for (int32_t j = 0; j < BLOCK_SIZE_A; j++)
		{
			colAData.push_back(i + j);
		}

		blockA.InsertData(colAData);
	}

	for (int32_t i = 0; i < BLOCK_COUNT_B; i++)
	{
		auto& blockB = colB.AddBlock();

		std::vector<int32_t> colBData;
		for (int32_t j = 0; j < BLOCK_SIZE_B; j++)
		{
			colBData.push_back(i + j);
		}

		blockB.InsertData(colBData);
	}

	// Perform the merge join
	auto start = std::chrono::steady_clock::now();

	MergeJoin::JoinUnique(colA, colB);

	auto end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    FAIL();
}