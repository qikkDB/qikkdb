#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <algorithm>
#include <iostream>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
    constexpr int32_t A_size = 8;
    constexpr int32_t B_size = 4;
    constexpr int32_t diag_count = A_size + B_size - 1;

    int32_t A[A_size] = {0, 0, 0, 1, 2, 2, 3, 4};
    int32_t B[B_size] = {0, 1, 2, 3};

    int32_t A_diag[diag_count];
    int32_t B_diag[diag_count];

    for (int32_t i = 0; i < diag_count; i++)
    {
        int32_t a_beg = std::max(0, i - B_size + 1);
        int32_t a_end = std::min(i, A_size - 1);

        int32_t b_beg = std::max(0, i - A_size + 1);
        int32_t b_end = std::min(i, B_size - 1);

        do
        {
            int32_t a_mid = a_beg + (a_end - a_beg) / 2;
            int32_t b_mid = b_end - (b_end - b_beg) / 2;

            if (A[a_mid] == B[b_mid])
            {
                A_diag[i] = a_mid;
                B_diag[i] = b_mid;

				std::printf("%d %d\n", a_mid, b_mid);
                break;
            } 
			else if(A[a_mid] < B[b_mid])
            {
                a_beg = a_mid + 1;
                b_end = b_mid - 1;
            }
            else if (A[a_mid] > B[b_mid])
            {
                a_end = a_mid - 1;
                b_beg = b_mid + 1;
            }
        } while (a_end >= a_beg && b_end >= b_beg);
    }

    FAIL();
}