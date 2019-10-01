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

TEST(GPUMergeJoinTests, MergeJoinCPUTest)
{
	constexpr int32_t A_size = 10;
	constexpr int32_t B_size = 8;

	constexpr int32_t W = 3;
	constexpr int32_t A_size_rounded = ((A_size + W - 1) / W) * W;
	constexpr int32_t B_size_rounded  = ((B_size + W - 1) / W) * W;

    constexpr int32_t diag_size_rounded = ((A_size_rounded + B_size_rounded - 1) / W ) * W;

	std::printf("%3d : %3d : %3d\n",A_size_rounded, B_size_rounded, diag_size_rounded );

    for (int32_t i = 0; i < diag_size_rounded; i++)
    {
        int32_t a_beg = i < B_size_rounded ? i % W : W + i - B_size_rounded;
        int32_t a_end = i < A_size_rounded ? i : A_size_rounded - W + i % W;

        int32_t b_beg = i < A_size_rounded ? (W - i % W - 1) : ((W + i - A_size_rounded) / W) * W + (W - i % W - 1);
        int32_t b_end = i < B_size_rounded ? (i / W) * W + (W - i % W - 1) : B_size_rounded - W + (W - i % W - 1);

        std::printf("%3d : %3d %3d %3d %3d\n", i, a_end, b_beg, a_beg, b_end);
    }

    FAIL();
}