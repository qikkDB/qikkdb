#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"
#include "../dropdbase/QueryEngine/CPUJoinReorderer.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
    // Initialize test buffers
    const int32_t BLOCK_SIZE_A = 13;
    const int32_t BLOCK_SIZE_B = 8;

    ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A, false, false);
    ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B, false, true);

    auto& blockA = colA.AddBlock();
    auto& blockB = colB.AddBlock();

    std::vector<int32_t> colAData = {'b', 'c', 'g', 'c', 'd', 'a', 'e',
                                     'a', 'e', 'c', 'f', 'g', 'c'};
    std::vector<int32_t> colBData = {'d', 'g', 'e', 'b', 'f', 'h', 'c', 'a'};

    blockA.InsertData(colAData);
    blockB.InsertData(colBData);

    // Initialize th output buffers
    std::vector<std::vector<int32_t>> colAJoinIndices;
    std::vector<std::vector<int32_t>> colBJoinIndices;

    // Perform the merge join
    // auto start = std::chrono::steady_clock::now();

    MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

    // auto end = std::chrono::steady_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

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

TEST(GPUMergeJoinTests, MergeJoinReorderTest)
{
    // Initialize test buffers
    const int32_t BLOCK_COUNT_A = 10;
    const int32_t BLOCK_SIZE_A = 1 << 10;

    const int32_t BLOCK_COUNT_B = 10;
    const int32_t BLOCK_SIZE_B = 1 << 10;

    ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A, false, false);
    ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B, false, true);

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

    // Initialize th output buffers
    std::vector<std::vector<int32_t>> colAJoinIndices;
    std::vector<std::vector<int32_t>> colBJoinIndices;

    // Perform the merge join
    MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

    // Reordered data
    std::vector<int32_t> colAReordererd;
    int32_t colAReordererdSize;

    std::vector<int32_t> colBReordererd;
    int32_t colBReordererdSize;

    // Reorder based on the join indices
    // auto start = std::chrono::steady_clock::now();

    for (int32_t i = 0; i < colA.GetBlockCount(); i++)
    {
        CPUJoinReorderer::reorderByJI(colAReordererd, colAReordererdSize, colA, i, colAJoinIndices, BLOCK_SIZE_A);
        CPUJoinReorderer::reorderByJI(colBReordererd, colBReordererdSize, colB, i, colBJoinIndices, BLOCK_SIZE_B);

        for (int32_t j = 0; j < colAReordererd.size(); j++)
        {
            ASSERT_EQ(colAReordererd[j], colBReordererd[j]);
        }
    }

    // auto end = std::chrono::steady_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // FAIL();
}

TEST(GPUMergeJoinTests, MergeJoinNULLTest)
{
    // Initialize test buffers
    const int32_t BLOCK_SIZE_A = 13;
    const int32_t BLOCK_SIZE_B = 8;

    ColumnBase<int32_t> colA("ColA", BLOCK_SIZE_A, true, false);
    ColumnBase<int32_t> colB("ColB", BLOCK_SIZE_B, false, true);

    auto& blockA = colA.AddBlock();
    auto& blockB = colB.AddBlock();

    std::vector<int32_t> colAData = {'b', 'c', 'g', 'c', 'd', 'a', 'e',
                                     'a', 'e', 'c', 'f', 'g', 'c'};
    std::vector<int32_t> colBData = {'d', 'g', 'e', 'b', 'f', 'h', 'c', 'a'};

    std::vector<nullmask_t> colANullMask = {static_cast<nullmask_t>(0b0000011101000011)};

    blockA.InsertData(colAData);
    blockB.InsertData(colBData);

    blockA.SetNullBitmask(colANullMask);

    // Initialize th output buffers
    std::vector<std::vector<int32_t>> colAJoinIndices;
    std::vector<std::vector<int32_t>> colBJoinIndices;

    // Perform the merge join
    MergeJoin::JoinUnique(colAJoinIndices, colBJoinIndices, colA, colB);

    std::vector<int32_t> colAJoinIndicesCorrect = {5, 7, 3, 12, 4, 2, 11};
    std::vector<int32_t> colBJoinIndicesCorrect = {7, 7, 6, 6, 0, 1, 1};

    ASSERT_EQ(colAJoinIndicesCorrect.size(), colAJoinIndices[0].size());
    ASSERT_EQ(colBJoinIndicesCorrect.size(), colBJoinIndices[0].size());
    for (int32_t i = 0; i < colAJoinIndices[0].size(); i++)
    {
        ASSERT_EQ(colAJoinIndicesCorrect[i], colAJoinIndices[0][i]);
        ASSERT_EQ(colBJoinIndicesCorrect[i], colBJoinIndices[0][i]);
    }
}