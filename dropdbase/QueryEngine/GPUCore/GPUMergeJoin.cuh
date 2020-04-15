#pragma once

/*
 * The merge join algorithm is based on the idea of the merge path with additional
 * efficient load balancing.
 * Used paper: GPU merge path: a GPU merging algorithm, O Green, R McColl, DA Bader 2012
 */

#include <cstdint>

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"

#include "../../ColumnBase.h"
#include "../../BlockBase.h"

#include "../../../cub/cub.cuh"

__global__ void kernel_label_input(int32_t* colBlockIndices, int32_t blockOffset, int32_t dataElementCount);

template <typename T>
__global__ void kernel_partition_input(int32_t* diagonalAIndices,
                                       int32_t* diagonalBIndices,
                                       T* colABlock,
                                       T* colBBlock,
                                       int32_t colABlockSize,
                                       int32_t colBBlockSize,
                                       int32_t colABlockSizeRounded,
                                       int32_t colBBlockSizeRounded,
                                       int32_t diagonalCount)
{

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < diagonalCount; i += stride)
    {
        const int32_t W = blockDim.x;
        const int32_t diagIdx = i / W;

        int32_t aBeg = i < colBBlockSizeRounded ? i % W : W + i - colBBlockSizeRounded;
        int32_t aEnd = i < colABlockSizeRounded ? i : colABlockSizeRounded - W + i % W;

        int32_t bBeg = i < colABlockSizeRounded ?
                           (W - i % W - 1) :
                           ((W + i - colABlockSizeRounded) / W) * W + (W - i % W - 1);
        int32_t bEnd = i < colBBlockSizeRounded ? (i / W) * W + (W - i % W - 1) :
                                                  colBBlockSizeRounded - W + (W - i % W - 1);

        // The merge condition is M[i] = A[a_i] > B[b_i]
        while (aBeg <= aEnd && bBeg <= bEnd)
        {
            int32_t aMid = aBeg + (aEnd - aBeg) / 2;
            int32_t bMid = bEnd - (bEnd - bBeg) / 2;

            // Check if the calcualted indices are within merge matrix bounds
            if (aMid >= colABlockSize && bMid >= colBBlockSize)
            {
                break;
            }
            else if (aMid >= colABlockSize && bMid < colBBlockSize)
            {
                aEnd = aMid - 1;
                bBeg = bMid + 1;

                continue;
            }
            else if (aMid < colABlockSize && bMid >= colBBlockSize)
            {
                aBeg = aMid + 1;
                bEnd = bMid - 1;

                continue;
            }

            // If this is a 1 and on the uppermost row or rightmost column, it is automatically a merge point
            if (colABlock[aMid] > colBBlock[bMid] && (aMid == 0 || bMid == (colBBlockSize - 1)))
            {
                diagonalAIndices[diagIdx] = aMid;
                diagonalBIndices[diagIdx] = bMid;

                break;
            }

            // If this is a 0 and on the lowermost row or leftmost column, it is automatically a merge point
            if (colABlock[aMid] <= colBBlock[bMid] && (aMid == (colABlockSize - 1) || bMid == 0))
            {
                diagonalAIndices[diagIdx] = aMid;
                diagonalBIndices[diagIdx] = bMid;

                break;
            }

            // Check merge point condition according to paper
            if (colABlock[aMid] > colBBlock[bMid - 1])
            {
                if (colABlock[aMid - 1] <= colBBlock[bMid])
                {
                    diagonalAIndices[diagIdx] = aMid;
                    diagonalBIndices[diagIdx] = bMid;

                    break;
                }
                else
                {
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


template <typename T>
__global__ void kernel_find_merge_path(int32_t* mergeAIndices,
                                       int32_t* mergeBIndices,
                                       int32_t* diagonalAIndices,
                                       int32_t* diagonalBIndices,
                                       T* colABlock,
                                       T* colBBlock,
                                       int32_t colABlockSize,
                                       int32_t colBBlockSize,
                                       int32_t diagonalCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < diagonalCount; i += stride)
    {
        const int32_t W = blockDim.x;
        const int32_t diagIdx = (i + 1) / W;

        const int32_t baseIdxA = (diagIdx == 0) ? 0 : diagonalAIndices[diagIdx - 1];
        const int32_t baseIdxB = (diagIdx == 0) ? 0 : diagonalBIndices[diagIdx - 1];

        int32_t aBegSub = baseIdxA;
        int32_t aEndSub = baseIdxA + ((diagIdx == 0) ? i % W : (i + 1) % W);

        int32_t bBegSub = baseIdxB;
        int32_t bEndSub = baseIdxB + ((diagIdx == 0) ? i % W : (i + 1) % W);

        while (aBegSub <= aEndSub && bBegSub <= bEndSub)
        {
            int32_t aMidSub = aBegSub + (aEndSub - aBegSub) / 2;
            int32_t bMidSub = bEndSub - (bEndSub - bBegSub) / 2;

            // Check if the calcualted indices are within merge matrix bounds
            if (aMidSub >= colABlockSize && bMidSub >= colBBlockSize)
            {
                break;
            }
            else if (aMidSub >= colABlockSize && bMidSub < colBBlockSize)
            {
                aEndSub = aMidSub - 1;
                bBegSub = bMidSub + 1;

                continue;
            }
            else if (aMidSub < colABlockSize && bMidSub >= colBBlockSize)
            {
                aBegSub = aMidSub + 1;
                bEndSub = bMidSub - 1;

                continue;
            }

            // If this is a 1 and on the uppermost row or rightmost column, it is automatically a merge point
            if (colABlock[aMidSub] > colBBlock[bMidSub] && (aMidSub == 0 || bMidSub == (colBBlockSize - 1)))
            {
                mergeAIndices[aMidSub + bMidSub] = aMidSub;
                mergeBIndices[aMidSub + bMidSub] = bMidSub;

                break;
            }

            // If this is a 0 and on the lowermost row or leftmost column, it is automatically a merge point
            if (colABlock[aMidSub] <= colBBlock[bMidSub] && (aMidSub == (colABlockSize - 1) || bMidSub == 0))
            {
                mergeAIndices[aMidSub + bMidSub] = aMidSub;
                mergeBIndices[aMidSub + bMidSub] = bMidSub;

                break;
            }

            // Check merge point condition according to paper
            if (colABlock[aMidSub] > colBBlock[bMidSub - 1])
            {
                if (colABlock[aMidSub - 1] <= colBBlock[bMidSub])
                {
                    mergeAIndices[aMidSub + bMidSub] = aMidSub;
                    mergeBIndices[aMidSub + bMidSub] = bMidSub;

                    break;
                }
                else
                {
                    aEndSub = aMidSub - 1;
                    bBegSub = bMidSub + 1;
                }
            }
            else
            {
                aBegSub = aMidSub + 1;
                bEndSub = bMidSub - 1;
            }
        }
    }
}

template <typename T>
__global__ void kernel_eval_predicate_merge_path(int8_t* joinPredicateMask,
                                                 int32_t* mergeAIndices,
                                                 int32_t* mergeBIndices,
                                                 T* colABlock,
                                                 T* colBBlock,
                                                 int32_t* colABlockIndices,
                                                 int64_t* colABlockNullMask,
                                                 int32_t diagonalCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < diagonalCount; i += stride)
    {
        // Evaluate the join predicate -> equijoin, consider the null values too
        if (colABlockNullMask)
        {
            // Fetch the null flag
            const bool nullFlag = static_cast<int32_t>(static_cast<uint8_t>(
                NullValues::GetConcreteBitFromBitmask(colABlockNullMask, colABlockIndices[mergeAIndices[i]])));


            // Evaluate the join condition
            joinPredicateMask[i] =
                !nullFlag && (colABlock[mergeAIndices[i]] == colBBlock[mergeBIndices[i]]);
        }
        else
        {
            // Evaluate the join condition
            joinPredicateMask[i] = (colABlock[mergeAIndices[i]] == colBBlock[mergeBIndices[i]]);
        }
    }
}

__global__ void kernel_compress_join_indices(int32_t* colABlockJoinIndices,
                                             int32_t* colBBlockJoinIndices,
                                             int8_t* joinPredicateMask,
                                             int32_t* joinPredicateMaskPSI,
                                             int32_t* mergeAIndices,
                                             int32_t* mergeBIndices,
                                             int32_t* colABlockIndices,
                                             int32_t* colBBlockIndices,
                                             int32_t diagonalCount);

class MergeJoin
{
public:
    // Column B must be unique !!!
    template <typename T>
    static void JoinUnique(std::vector<std::vector<int32_t>>& colAJoinIndices,
                           std::vector<std::vector<int32_t>>& colBJoinIndices,
                           ColumnBase<T>& colA,
                           ColumnBase<T>& colB)
    {
        // Fetch the context
        Context& context = Context::getInstance();

        // Clear the input vectors
        colAJoinIndices.clear();
        colBJoinIndices.clear();

        // Fetch the column properties for both A and B columns
        const auto colABlockList = colA.GetBlocksList();
        const int32_t colABlockCapacity = colA.GetBlockSize();
        const int32_t colABlockCount = colA.GetBlockCount();

        const auto colBBlockList = colB.GetBlocksList();
        const int32_t colBBlockCapacity = colB.GetBlockSize();
        const int32_t colBBlockCount = colB.GetBlockCount();

        const bool colAisNullable = colA.GetIsNullable();
        const bool colBisNullable = colB.GetIsNullable();

        const bool colAisUnique = colA.GetIsUnique();
        const bool colBisUnique = colB.GetIsUnique();

        // Check for zero block capacity
        if (colABlockCapacity == 0 || colBBlockCapacity == 0)
        {
            // TODO Handle
            return;
        }

        // Check for collumnt uniqueness
        if (!colBisUnique)
        {
            throw std::runtime_error("[ERROR] Column B no unique\n");
        }

        // Calculate the  merge path diagonal work buffer sizes
        const int32_t W = context.getBlockDim();
        const int32_t colABlockCapacityRounded = ((colABlockCapacity + W - 1) / W) * W;
        const int32_t colBBlockCapacityRounded = ((colBBlockCapacity + W - 1) / W) * W;

        const int32_t diagonalCountCapacityRounded =
            (colABlockCapacityRounded + colBBlockCapacityRounded - 1);
        const int32_t diagonalCountSparseCapacityRounded =
            (colABlockCapacityRounded + colBBlockCapacityRounded - 1) / W;

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
        cub::DeviceRadixSort::SortPairs(nullptr, tempStorageSizeA, colABlock.get(),
                                        colABlockSorted.get(), colABlockIndices.get(),
                                        colABlockIndicesSorted.get(), colABlockCapacity);

        size_t tempStorageSizeB = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tempStorageSizeB, colBBlock.get(),
                                        colBBlockSorted.get(), colBBlockIndices.get(),
                                        colBBlockIndicesSorted.get(), colBBlockCapacity);

        cuda_ptr<int8_t> tempStorageA(tempStorageSizeA);
        cuda_ptr<int8_t> tempStorageB(tempStorageSizeB);

        // Alloc the merge path diagonal intersections work buffers
        cuda_ptr<int32_t> diagonalAIndices(diagonalCountSparseCapacityRounded);
        cuda_ptr<int32_t> diagonalBIndices(diagonalCountSparseCapacityRounded);

        // Alloc the merge path indices buffers
        cuda_ptr<int32_t> mergeAIndices(diagonalCountCapacityRounded);
        cuda_ptr<int32_t> mergeBIndices(diagonalCountCapacityRounded);

        GPUMemory::memset(mergeAIndices.get(), 0, diagonalCountCapacityRounded);

        // Alloc the join predicate mask
        cuda_ptr<int8_t> joinPredicateMask(diagonalCountCapacityRounded);

        // Alloc the join predicate mask prefix sum buffer and the temporary storage
        cuda_ptr<int32_t> joinPredicateMaskPSI(diagonalCountCapacityRounded);

        size_t tempStorageSizePSI = 0;
        cub::DeviceScan::InclusiveSum(nullptr, tempStorageSizePSI, joinPredicateMask.get(),
                                      joinPredicateMaskPSI.get(), diagonalCountCapacityRounded);

        cuda_ptr<int8_t> tempStoragePSI(tempStorageSizePSI);

        // Alloc the join indices buffer
        cuda_ptr<int32_t> colABlockJoinIndices(diagonalCountCapacityRounded);
        cuda_ptr<int32_t> colBBlockJoinIndices(diagonalCountCapacityRounded);

        // Calculate the null block size and alloc the null mask buffers
        size_t colABlockNullMaskCapacity = NullValues::GetNullBitMaskSize(colABlockCapacity);

        cuda_ptr<int64_t> colABlockNullMask(colABlockNullMaskCapacity);

        // Perform the merge join
        for (int32_t a = 0; a < colABlockCount; a++)
        {
            // Fetch the A block size and null mask size
            const int32_t colABlockSize = colABlockList[a]->GetSize();
            const int32_t colABlockNullMaskSize = NullValues::GetNullBitMaskSize(colABlockSize);
            if (colABlockSize == 0)
            {
                continue;
            }

            // Label the input indices and copy the input join keys to the GPU for the A block
            kernel_label_input<<<context.calcGridDim(colABlockSize), context.getBlockDim()>>>(
                colABlockIndices.get(), a * colABlockCapacity, colABlockSize);
            CheckCudaError(cudaGetLastError());

            // Copy the block content to the gpu, if the null mask is present copy it aswell
            GPUMemory::copyHostToDevice(colABlock.get(), colABlockList[a]->GetData(), colABlockSize);

            int64_t* d_colABlockNullMask = nullptr;
            if (colAisNullable && colABlockList[a]->GetNullBitmask())
            {
                GPUMemory::copyHostToDevice(colABlockNullMask.get(),
                                            colABlockList[a]->GetNullBitmask(), colABlockNullMaskSize);

                d_colABlockNullMask = colABlockNullMask.get();
            }

            // Sort the input based on the join keys for the A block
            cub::DeviceRadixSort::SortPairs(tempStorageA.get(), tempStorageSizeA, colABlock.get(),
                                            colABlockSorted.get(), colABlockIndices.get(),
                                            colABlockIndicesSorted.get(), colABlockSize);

            for (int32_t b = 0; b < colBBlockCount; b++)
            {
                // Fetch the B block size
                const int32_t colBBlockSize = colBBlockList[b]->GetSize();
                if (colBBlockSize == 0)
                {
                    continue;
                }

                // Label the input indices and copy the input join keys to the GPU for the B block
                kernel_label_input<<<context.calcGridDim(colBBlockSize), context.getBlockDim()>>>(
                    colBBlockIndices.get(), b * colBBlockCapacity, colBBlockSize);
                CheckCudaError(cudaGetLastError());

                GPUMemory::copyHostToDevice(colBBlock.get(), colBBlockList[b]->GetData(), colBBlockSize);

                // Sort the input based on the join keys for the A block
                cub::DeviceRadixSort::SortPairs(tempStorageB.get(), tempStorageSizeB, colBBlock.get(),
                                                colBBlockSorted.get(), colBBlockIndices.get(),
                                                colBBlockIndicesSorted.get(), colBBlockSize);

                // Find merge path diagonal intersections
                const int32_t colABlockSizeRounded = ((colABlockSize + W - 1) / W) * W;
                const int32_t colBBlockSizeRounded = ((colBBlockSize + W - 1) / W) * W;

                const int32_t diagonalCountRounded = colABlockSizeRounded + colBBlockSizeRounded - 1;

                kernel_partition_input<<<context.calcGridDim(diagonalCountRounded), W>>>(
                    diagonalAIndices.get(), diagonalBIndices.get(), colABlockSorted.get(),
                    colBBlockSorted.get(), colABlockSize, colBBlockSize, colABlockSizeRounded,
                    colBBlockSizeRounded, diagonalCountRounded);
                CheckCudaError(cudaGetLastError());

                // Calculate the real diagonal count
                const int32_t diagonalCount = (colABlockSize + colBBlockSize - 1);

                // Merge the two arrays - find the merge path
                kernel_find_merge_path<<<context.calcGridDim(diagonalCount), W>>>(
                    mergeAIndices.get(), mergeBIndices.get(), diagonalAIndices.get(),
                    diagonalBIndices.get(), colABlockSorted.get(), colBBlockSorted.get(),
                    colABlockSize, colBBlockSize, diagonalCount);
                CheckCudaError(cudaGetLastError());

                // Zero the predicate mask
                GPUMemory::memset(joinPredicateMask.get(), 0, diagonalCountCapacityRounded);

                // Evaluate the join predicate on the merge path
                kernel_eval_predicate_merge_path<<<context.calcGridDim(diagonalCount), W>>>(
                    joinPredicateMask.get(), mergeAIndices.get(), mergeBIndices.get(),
                    colABlockSorted.get(), colBBlockSorted.get(), colABlockIndicesSorted.get(),
                    d_colABlockNullMask, diagonalCount);
                CheckCudaError(cudaGetLastError());

                // Zero the prefix sum
                GPUMemory::memset(joinPredicateMaskPSI.get(), 0, diagonalCountCapacityRounded);

                // Run the prefix sum on the join predicate mask
                cub::DeviceScan::InclusiveSum(tempStoragePSI.get(), tempStorageSizePSI,
                                              joinPredicateMask.get(), joinPredicateMaskPSI.get(),
                                              diagonalCountCapacityRounded);
                CheckCudaError(cudaGetLastError());

                // Fetch the final size of the matching join conditions
                int32_t joinMatchCount;
                GPUMemory::copyDeviceToHost(&joinMatchCount,
                                            joinPredicateMaskPSI.get() + diagonalCountCapacityRounded - 1, 1);

                // Compress the join indices
                kernel_compress_join_indices<<<context.calcGridDim(diagonalCount), W>>>(
                    colABlockJoinIndices.get(), colBBlockJoinIndices.get(), joinPredicateMask.get(),
                    joinPredicateMaskPSI.get(), mergeAIndices.get(), mergeBIndices.get(),
                    colABlockIndicesSorted.get(), colBBlockIndicesSorted.get(), diagonalCountCapacityRounded);
                CheckCudaError(cudaGetLastError());

                // Copy the partially joined tuple indices back to the cpu RAM
                colAJoinIndices.push_back(std::vector<int32_t>(joinMatchCount));
                colBJoinIndices.push_back(std::vector<int32_t>(joinMatchCount));

                GPUMemory::copyDeviceToHost(colAJoinIndices.back().data(),
                                            colABlockJoinIndices.get(), joinMatchCount);
                GPUMemory::copyDeviceToHost(colBJoinIndices.back().data(),
                                            colBBlockJoinIndices.get(), joinMatchCount);
            }
        }
    }
};