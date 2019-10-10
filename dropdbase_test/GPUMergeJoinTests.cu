#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
	// Initialize test buffers
    const int32_t BLOCK_COUNT_A = 13; // 1;
    const int32_t BLOCK_SIZE_A = 1 << 25; // 13;	

	const int32_t BLOCK_COUNT_B = 13; // 1;
    const int32_t BLOCK_SIZE_B = 1 << 25; // 8;	

	ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A);
	ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B);
	
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
	
    
	/*
    auto& blockA = colA.AddBlock();
    auto& blockB = colB.AddBlock();

    std::vector<int32_t> colAData = {'b', 'c', 'g', 'c', 'd', 'a', 'e',
                                     'a', 'e', 'c', 'f', 'g', 'c'
    };
    std::vector<int32_t> colBData = {
        'd', 'g', 'e', 'b', 'f', 'h', 'c', 'a'
    };

	blockA.InsertData(colAData);
    blockB.InsertData(colBData);
	*/

	// Initialize th output buffers
    std::vector<int32_t> colAJoinIndices;
    std::vector<int32_t> colBJoinIndices;

	// Perform the merge join
	auto start = std::chrono::steady_clock::now();

	MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

	auto end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	// DEBUG
    /*
    for (int32_t x = 0; x < colAJoinIndices.size(); x++)
    {
        std::printf("%3d : %3d %3d\n", x , colAJoinIndices[x], colBJoinIndices[x]);
	}
	*/

    FAIL();
}