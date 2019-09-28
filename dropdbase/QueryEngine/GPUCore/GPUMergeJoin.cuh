#pragma once

#include <cstdint>

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"

#include "../../ColumnBase.h"
#include "../../BlockBase.h"

#include "../../../cub/cub.cuh"

/*
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


		// Check merge point condition according to paper
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
*/

__global__ void kernel_label_input(int32_t *colBlockIndices, int32_t blockOffset, int32_t dataElementCount);

template<typename T> 
__global__ void kernel_partition_input(int32_t diagonalCount) 
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < diagonalCount; i += stride)
	{

	}
}

class MergeJoin
{
public:
	// Column B is considered unique
    template<typename T> 
	static void JoinUnique(ColumnBase<T>& colA, ColumnBase<T>& colB)
    {
		// Fetch the column properties for both A and B columns
		const auto colABlockList = colA.GetBlocksList();
		const int32_t colABlockCapacity = colA.GetBlockSize();
		const int32_t colABlockCount = colA.GetBlockCount();

		const auto colBBlockList = colB.GetBlocksList();
		const int32_t colBBlockCapacity = colB.GetBlockSize();
		const int32_t colBBlockCount = colB.GetBlockCount();

		// Check for zero block capacity
		if(colABlockCapacity == 0 || colBBlockCapacity == 0) {
			// TODO Handle
			return;
		}

		// Alloc the work buffers for both A and B blocks
		cuda_ptr<T> colABlock(colABlockCapacity);
		cuda_ptr<int32_t> colABlockIndices(colABlockCapacity);

		cuda_ptr<T> colABlockSorted(colABlockCapacity);
		cuda_ptr<int32_t> colABlockIndicesSorted(colABlockCapacity);

		cuda_ptr<T> colBBlock(colBBlockCapacity);
		cuda_ptr<int32_t> colBBlockIndices(colBBlockCapacity);

		cuda_ptr<T> colBBlockSorted(colBBlockCapacity);
		cuda_ptr<int32_t> colBBlockIndicesSorted(colBBlockCapacity);

		// Alloc the radix sort buffers
		size_t tempStorageSizeA = 0;
		cub::DeviceRadixSort::SortPairs(nullptr, tempStorageSizeA, colABlock.get(), colABlockSorted.get(), colABlockIndices.get(), colABlockIndicesSorted.get(), colABlockCapacity);

		size_t tempStorageSizeB = 0;
		cub::DeviceRadixSort::SortPairs(nullptr, tempStorageSizeB, colBBlock.get(), colBBlockSorted.get(), colBBlockIndices.get(), colBBlockIndicesSorted.get(), colBBlockCapacity);
			
		cuda_ptr<int8_t> tempStorageA(tempStorageSizeA);
		cuda_ptr<int8_t> tempStorageB(tempStorageSizeB);

		// Perform the merge join
        for (int32_t a = 0; a < colABlockCount; a++)
        {
			// Fetch the A block size
            int32_t colABlockSize = colABlockList[a]->GetSize();
			if(colABlockSize == 0){
				continue;
			}

			// Label the input indices and copy the input join keys to the GPU for the A block
			kernel_label_input<<<Context::getInstance().calcGridDim(colABlockSize), 
								 Context::getInstance().getBlockDim()>>>(colABlockIndices.get(), a * colABlockCapacity, colABlockSize);
            GPUMemory::copyHostToDevice(colABlock.get(), colABlockList[a]->GetData(), colABlockSize);

			// Sort the input based on the join keys for the A block
			cub::DeviceRadixSort::SortPairs(tempStorageA.get(), tempStorageSizeA, colABlock.get(), colABlockSorted.get(), colABlockIndices.get(), colABlockIndicesSorted.get(), colABlockSize);

            for (int32_t b = 0; b < colBBlockCount; b++)
            {
				// Fetch the B block size
				int32_t colBBlockSize = colBBlockList[b]->GetSize();
				if(colBBlockSize == 0){
					continue;
				}

				// Label the input indices and copy the input join keys to the GPU for the B block
				kernel_label_input<<<Context::getInstance().calcGridDim(colBBlockSize), 
						Context::getInstance().getBlockDim()>>>(colBBlockIndices.get(), b * colBBlockCapacity, colBBlockSize);
                GPUMemory::copyHostToDevice(colBBlock.get(), colBBlockList[b]->GetData(), colBBlockSize);

				// Sort the input based on the join keys for the A block
				cub::DeviceRadixSort::SortPairs(tempStorageB.get(), tempStorageSizeB, colBBlock.get(), colBBlockSorted.get(), colBBlockIndices.get(), colBBlockIndicesSorted.get(), colBBlockSize);
			}
		}
	}

	template<typename T> 
	static void Join(ColumnBase<T>& colA, ColumnBase<T>& colB)
    {
		// TODO
	}
};