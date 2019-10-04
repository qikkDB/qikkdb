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

	// Perform the merge join
	auto start = std::chrono::steady_clock::now();

	MergeJoin::JoinUnique(colA, colB);

	auto end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    FAIL();
}

TEST(GPUMergeJoinTests, SimpleMerge)
{
    std::vector<int32_t> A = {'a', 'a', 'b', 'c', 'c', 'c', 'c',
                                     'd', 'e', 'e', 'f', 'g', 'g'};
    std::vector<int32_t> B = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	std::vector<int32_t> C(A.size() + B.size());

	 int a = 0, b = 0, c = 0;

    // Classic merge
     while (a < A.size() && b < B.size())
    {
        if (A[a] > B[b])
        {
            printf("CB: %3d : %3d %3d\n", c, a, b);

            C[c++] = B[b++];
        }
        else if (A[a] < B[b])
        {
            printf("CA: %3d : %3d %3d\n", c, a, b);

            C[c++] = A[a++];
        }
        else if (A[a] == B[b])
        {
            printf("CA: %3d : %3d %3d\n", c, a, b);

            C[c++] = A[a++]; // If B has unique keys
            // C[c++] = B[b++];	// If A has unique keys
        }
    }

    while (a < A.size())
    {
        printf("A : %3d : %3d %3d\n", c, a, b);

        C[c++] = A[a++];
    }

    while (b < B.size())
    {
        printf("B : %3d : %3d %3d\n", c, a, b);

        C[c++] = B[b++];
    }

	FAIL();
}