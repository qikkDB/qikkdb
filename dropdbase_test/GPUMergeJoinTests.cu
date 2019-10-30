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

    auto& blockA = colA.AddBlock();
    auto& blockB = colB.AddBlock();

    std::vector<int32_t> colAData = {'b', 'c', 'g', 'c', 'd', 'a', 'e', 'a', 'e', 'c', 'f', 'g', 'c'};
    std::vector<int32_t> colBData = {'d', 'g', 'e', 'b', 'f', 'h', 'c', 'a'};

	blockA.InsertData(colAData);
    blockB.InsertData(colBData);

	// Initialize th output buffers
    std::vector<std::vector<int32_t>> colAJoinIndices;
    std::vector<std::vector<int32_t>> colBJoinIndices;

	// Perform the merge join
	//auto start = std::chrono::steady_clock::now();

	MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

	//auto end = std::chrono::steady_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	std::vector<int32_t> colAJoinIndicesCorrect = {5, 7, 0, 1, 3, 9, 12, 4, 6, 8, 10, 2, 11};
    std::vector<int32_t> colBJoinIndicesCorrect = {7, 7, 3, 6, 6, 6, 6, 0, 2, 2, 4, 1, 1};

	ASSERT_EQ(colAJoinIndices[0].size(), colAJoinIndicesCorrect.size());
    for (int32_t i = 0; i < colAJoinIndices[0].size(); i++)
    {
        ASSERT_EQ(colAJoinIndices[0][i], colAJoinIndicesCorrect[i]);
	}

	ASSERT_EQ(colBJoinIndices[0].size(), colBJoinIndicesCorrect.size());
    for (int32_t i = 0; i < colBJoinIndices[0].size(); i++)
    {
        ASSERT_EQ(colBJoinIndices[0][i], colBJoinIndicesCorrect[i]);
    }
}