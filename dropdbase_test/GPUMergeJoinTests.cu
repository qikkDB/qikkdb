#include "../dropdbase/QueryEngine/GPUCore/GPUMergeJoin.cuh"

#include "gtest/gtest.h"

#include <cstdint>
#include <algorithm>
#include <iostream>

TEST(GPUMergeJoinTests, MergeJoinTest)
{
    constexpr int32_t A_size = 8;
	constexpr int32_t B_size = 4;
    //constexpr int32_t B_size = 8;
    constexpr int32_t diag_size = A_size + B_size - 1;

    //int32_t A[A_size] = {17, 29, 35, 73, 86, 90, 95, 99};
    //int32_t B[B_size] = {3, 5, 12, 22, 45, 64, 69, 82};

	int32_t A[A_size] = {0, 0, 0, 1, 2, 2, 3, 4};
    int32_t B[B_size] = {0, 1, 2, 3};

    int32_t A_diag[diag_size];
    int32_t B_diag[diag_size];
	for (int32_t i = 0; i < diag_size; i++)
    {
		A_diag[i] = -1;
		B_diag[i] = -1;
	}

    for (int32_t i = 0; i < diag_size; i++)
    {
        int32_t a_beg = std::max(0, i - B_size + 1);
        int32_t a_end = std::min(i, A_size - 1);

        int32_t b_beg = std::max(0, i - A_size + 1);
        int32_t b_end = std::min(i, B_size - 1);

		// The merge condition is M[i] = A[a_i] > B[b_i]
        while (a_beg <= a_end && b_beg <= b_end)
        {
            int32_t a_mid = a_beg + (a_end - a_beg) / 2;
            int32_t b_mid = b_end - (b_end - b_beg) / 2;

			// If this is a 1 and on the uppermost row or rightmost column, it is automatically a merge point
			if(A[a_mid] > B[b_mid] && (a_mid == 0 || b_mid == (B_size - 1))) {
				A_diag[a_mid + b_mid] = a_mid;
				B_diag[a_mid + b_mid] = b_mid;

				break;
			}

			// If this is a 0 and on the lowermost row or leftmost column, it is automatically a merge point
			if(A[a_mid] < B[b_mid] && (a_mid == (A_size - 1) || b_mid == 0)) {
				A_diag[a_mid + b_mid] = a_mid;
				B_diag[a_mid + b_mid] = b_mid;
				break;
			}


            if (A[a_mid] > B[b_mid - 1])
            {
				if(A[a_mid - 1] <= B[b_mid]) {
					A_diag[a_mid + b_mid] = a_mid;
					B_diag[a_mid + b_mid] = b_mid;

					break;
				}
				else {
				a_end = a_mid - 1;
				b_beg = b_mid + 1;
				}
			}
			else
            {
                a_beg = a_mid + 1;
                b_end = b_mid - 1;
            }
        }
    }

	for (int32_t i = 0; i < diag_size; i++)
    {
		std::printf("%d : %d %d\n",i,  A_diag[i], B_diag[i]);
	}

    FAIL();
}