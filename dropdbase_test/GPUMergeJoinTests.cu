#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
	// Initialize test buffers
	const int32_t BLOCK_COUNT_A = 1;
    const int32_t BLOCK_SIZE_A = 13;	

	const int32_t BLOCK_COUNT_B = 1;
    const int32_t BLOCK_SIZE_B = 8;	

	ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A);
	ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B);

	/*
	for (int32_t i = 0; i < BLOCK_COUNT_A; i++)
	{
		auto& blockA = colA.AddBlock();

		std::vector<int32_t> colAData;
		for (int32_t j = 0; j < BLOCK_SIZE_A; j++)
		{
            colAData.push_back(i * BLOCK_SIZE_A + j);
		}

		blockA.InsertData(colAData);
	}

	for (int32_t i = 0; i < BLOCK_COUNT_B; i++)
	{
		auto& blockB = colB.AddBlock();

		std::vector<int32_t> colBData;
		for (int32_t j = 0; j < BLOCK_SIZE_B; j++)
		{
            colBData.push_back(i * BLOCK_SIZE_B + j);
		}

		blockB.InsertData(colBData);
	}
	*/
    
    auto& blockA = colA.AddBlock();
    auto& blockB = colB.AddBlock();

    std::vector<int32_t> colAData = {
        'a', 'a', 'b', 'c', 'c', 'c', 'c', 'd', 'e', 'e', 'f', 'g', 'g'
    };
    std::vector<int32_t> colBData = {
		'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'
    };

	blockA.InsertData(colAData);
    blockB.InsertData(colBData);

	// Initialize th output buffers
    std::vector<int32_t> colAJoinIndices;
    std::vector<int32_t> colBJoinIndices;

	// Perform the merge join
	auto start = std::chrono::steady_clock::now();

	MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

	auto end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    FAIL();
}