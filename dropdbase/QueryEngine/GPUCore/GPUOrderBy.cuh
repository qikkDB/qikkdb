#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <cstdint>
#include <iostream>

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"
#include "GPUReconstruct.cuh"
#include "../OrderByType.h"
#include "../../IVariantArray.h"
#include "cuda_ptr.h"
#include "IOrderBy.h"

#include "../../../cub/cub.cuh"

// Fill the index buffers with default indices
__global__ void kernel_fill_indices(int32_t* indices, int32_t dataElementCount);

// Transform the input data null rows to the smallest possible value
template <typename T>
__global__ void kernel_transform_null_values(T* inCol, int64_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Retrieve the null flag
        bool isNullFlag = NullValues::GetConcreteBitFromBitmask(nullBitMask, i);
        if (isNullFlag)
        {
            inCol[i] = std::numeric_limits<T>::lowest();
        }
    }
}

// Reorder a column by a given index column
template <typename T>
__global__ void kernel_reorder_by_idx(T* outCol, int32_t* inIndices, T* inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        outCol[i] = inCol[inIndices[i]];
    }
}

__global__ void kernel_reorder_chars_by_idx(GPUMemory::GPUString outCol,
                                            int32_t* inIndices,
                                            GPUMemory::GPUString inCol,
                                            int32_t* outStringIndices,
                                            int32_t* outStringLenghts,
                                            int32_t dataElementCount);

/// Reorder polygon points by already reordered polygon and point indices/lengths
__global__ void kernel_reorder_points_by_idx(GPUMemory::GPUPolygon outCol,
                                             int32_t* inIndices,
                                             GPUMemory::GPUPolygon inCol,
                                             int32_t* outPolygonIndices,
                                             int32_t* outPolygonLengths,
                                             int32_t* outPointIndices,
                                             int32_t* outPointLengths,
                                             int32_t dataElementCount);

/// Reorder pointIdx array of GPUPolygon struct based on already reordered polyCount and polyIdx arrays
__global__ void kernel_reorder_point_counts_by_poly_idx_lenghts(int32_t* outPointLengths,
                                                                int32_t* inOrderIndices,
                                                                int32_t* inPointLenghts,
                                                                int32_t* inPolygonIndices,
                                                                int32_t* outPolygonIndices,
                                                                int32_t* outPolygonLengths,
                                                                int32_t dataElementCount);

// Reorder a null column by a given index column
__global__ void kernel_reorder_null_values_by_idx(int64_t* outNullBitMask,
                                                  int32_t* inIndices,
                                                  int64_t* inNullBitMask,
                                                  int32_t dataElementCount);

class GPUOrderBy : public IOrderBy
{
private:
    // Radix indices front and back buffer
    int32_t* indices1;
    int32_t* indices2;

public:
    GPUOrderBy(int32_t dataElementCount);

    ~GPUOrderBy();

    // Set the null value rows to the smallest possible value for the given type for merge operations
    template <typename T>
    static void TransformNullValsToSmallestVal(T* inCol, int64_t* nullBitMask, int32_t dataElementCount)
    {
        if (nullBitMask != nullptr)
        {
            kernel_transform_null_values<<<Context::getInstance().calcGridDim(dataElementCount),
                                           Context::getInstance().getBlockDim()>>>(inCol, nullBitMask,
                                                                                   dataElementCount);
        }
    }

    // The base order by method for numeric types
    // Iterate trough all the columns and sort them with radix sort
    // Handle the columns as if they were a big binary number from right to left
    // If nullBitMask is nullptr - dont use null values
    // for(int32_t i = inCols.size() - 1; i >= 0; i--)
    template <typename T>
    void OrderByColumn(int32_t* outIndices, T* inCol, int64_t* nullBitMask, int32_t dataElementCount, OrderBy::Order order)
    {
        // Preprocess the columns with the null values
        TransformNullValsToSmallestVal(inCol, nullBitMask, dataElementCount);

        // Keys front and back buffer
        cuda_ptr<T> keys1(dataElementCount);
        cuda_ptr<T> keys2(dataElementCount);

        // Radix sort helper buffers - alloc them
        size_t radix_temp_buf_size_ = 0;
        int64_t* radix_temp_buf_ = nullptr;
        cub::DeviceRadixSort::SortPairs(radix_temp_buf_, radix_temp_buf_size_, keys1.get(),
                                        keys2.get(), indices1, indices2, dataElementCount);

        GPUMemory::alloc(&radix_temp_buf_, radix_temp_buf_size_);

        // Copy the keys to the first key buffer and
        // rotate the keys in the higher orders based on the
        // indices from all the lower radices
        ReOrderByIdx(keys1.get(), indices1, inCol, dataElementCount);

        // Perform radix sort
        // Ascending
        switch (order)
        {
        case OrderBy::Order::ASC:
            cub::DeviceRadixSort::SortPairs(radix_temp_buf_, radix_temp_buf_size_, keys1.get(),
                                            keys2.get(), indices1, indices2, dataElementCount);
            break;
        case OrderBy::Order::DESC:
            cub::DeviceRadixSort::SortPairsDescending(radix_temp_buf_, radix_temp_buf_size_, keys1.get(),
                                                      keys2.get(), indices1, indices2, dataElementCount);
            break;
        }

        // Swap GPU pointers
        int32_t* indices_temp = indices1;
        indices1 = indices2;
        indices2 = indices_temp;

        // Copy the resulting indices to the output
        GPUMemory::copyDeviceToDevice(outIndices, indices1, dataElementCount);

        GPUMemory::free(radix_temp_buf_);
    }

    template <typename T>
    static void ReOrderByIdx(T* outCol, int32_t* inIndices, T* inCol, int32_t dataElementCount)
    {
        // Reorder a column based on indices
        kernel_reorder_by_idx<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(outCol, inIndices, inCol, dataElementCount);
    }

    static void ReOrderStringByIdx(GPUMemory::GPUString& outCol,
                                   int32_t* inIndices,
                                   GPUMemory::GPUString inCol,
                                   int32_t dataElementCount);

    static void ReOrderPolygonByIdx(GPUMemory::GPUPolygon& outCol,
                                    int32_t* inIndices,
                                    GPUMemory::GPUPolygon inCol,
                                    int32_t dataElementCount);

    template <typename T>
    static void ReOrderByIdxInplace(T* col, int32_t* indices, int32_t dataElementCount)
    {
        cuda_ptr<T> outTemp(dataElementCount);
        GPUMemory::copyDeviceToDevice(outTemp.get(), col, dataElementCount);

        // Reorder a column based on indices "inplace"
        kernel_reorder_by_idx<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(col, indices, outTemp.get(), dataElementCount);
    }

    static void ReOrderNullValuesByIdx(int64_t* outNullBitMask, int32_t* indices, int64_t* inNullBitMask, int32_t dataElementCount);
};