#pragma once

#include <cstdint>

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"

#include "../../ColumnBase.h"
#include "../../BlockBase.h"

#include "../../../cub/cub.cuh"

__global__ void kernel_label_input(int32_t *colBlockIndices, int32_t blockOffset, int32_t dataElementCount);

template<typename T> 
__global__ void kernel_partition_input(int32_t *diagonalAIndices,
									   int32_t *diagonalBIndices,
									   T *colABlock, 
									   T *colBBlock, 
									   int32_t colABlockSize, 
									   int32_t colBBlockSize, 
									   int32_t diagonalCount) 
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < diagonalCount; i += stride)
	{
		int32_t aBeg = max(0, i - colBBlockSize + 1);
		int32_t aEnd = min(i, colABlockSize - 1);

		int32_t bBeg = max(0, i - colABlockSize + 1);
		int32_t bEnd = min(i, colBBlockSize - 1);

		// The merge condition is M[i] = A[a_i] > B[b_i]
		while (aBeg <= aEnd && bBeg <= bEnd)
		{
			int32_t aMid = aBeg + (aEnd - aBeg) / 2;
			int32_t bMid = bEnd - (bEnd - bBeg) / 2;

			// If this is a 1 and on the uppermost row or rightmost column, it is automatically a merge point
			if(colABlock[aMid] > colBBlock[bMid] && (aMid == 0 || bMid == (colBBlockSize - 1))) {
				diagonalAIndices[aMid + bMid] = aMid;
				diagonalBIndices[aMid + bMid] = bMid;

				break;
			}

			// If this is a 0 and on the lowermost row or leftmost column, it is automatically a merge point
			if(colABlock[aMid] < colBBlock[bMid] && (aMid == (colABlockSize - 1) || bMid == 0)) {
				diagonalAIndices[aMid + bMid] = aMid;
				diagonalBIndices[aMid + bMid] = bMid;
				break;
			}


			// Check merge point condition according to paper
			if (colABlock[aMid] > colBBlock[bMid - 1])
			{
				if(colABlock[aMid - 1] <= colBBlock[bMid]) {
					diagonalAIndices[aMid + bMid] = aMid;
					diagonalBIndices[aMid + bMid] = bMid;

					break;
				}
				else {
				aEnd = aMid - 1;
				bBeg = bMid + 1;
				}
			}
			else
			{
				aBeg = aMid + 1;
				bEnd = bMid - 1;
			}
		}
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
			
				// TODO Find merge path diagonal intersections

				// TODO Merge the two arrays - find join pairs				

			}
		}
	}

	template<typename T> 
	static void Join(ColumnBase<T>& colA, ColumnBase<T>& colB)
    {
		// TODO
	}
};