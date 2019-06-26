#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <cstdint>
#include <iostream>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"

#include "../../../cub/cub.cuh"

// Fill the index buffers with default indices
__global__ void kernel_fill_indices(int32_t* indices, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        indices[i] = i;
	}
}

// Reorder a column by a given index column
template<typename T>
__global__ void kernel_reorder_by_idx(T* outCol, int32_t* inIndices, T* inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        outCol[i] = inCol[inIndices[i]];
    }
}

template<typename T>
class GPUOrderBy {
private:
    // Radix indices front and back buffer
    int32_t* indices1;
    int32_t* indices2;

    // Keys front and back buffer
    T* keys1;
    T* keys2;

    // Radix sort helper buffers
    size_t radix_temp_buf_size_;
    int8_t* radix_temp_buf_;

    // The base order by method for numeric types
    void OrderBy(std::vector<int32_t*> &outIndices, std::vector<T*> &inCols, int32_t dataElementCount, bool ascending = true)
    {
        // Initialize the index buffer
        kernel_fill_indices<<< Context::getInstance().calcGridDim(dataElementCount), 
                               Context::getInstance().getBlockDim() >>>
                               (indices1, dataElementCount);

        // Iterate trough all the columns and sort them with radix sort
        // Handle the columns as if they were a big binary number from right to left
        for(int32_t i = inCols.size() - 1; i >= 0; i--)
        {
            // Copy the keys to the first key buffer
            GPUMemory::copyDeviceToDevice(keys1, inCols[i], dataElementCount);

            // Perform radix sort
            // Ascending
            if(ascending)
            {
                cub::DeviceRadixSort::SortPairs(radix_temp_buf_, 
                                                radix_temp_buf_size_,
                                                keys1, 
                                                keys2, 
                                                indices1, 
                                                indices2,
                                                dataElementCount); 
            }
            else
            {
                cub::DeviceRadixSort::SortPairsDescending(radix_temp_buf_, 
                                                          radix_temp_buf_size_,
                                                          keys1, 
                                                          keys2, 
                                                          indices1, 
                                                          indices2,
                                                          dataElementCount);
            }

            // Copy the resulting indices to the output
            GPUMemory::copyDeviceToDevice(outIndices[i], indices2, dataElementCount);

            // Swap GPU pointers
            int32_t* indices_temp = indices1;
            indices1 = indices2;
            indices2 = indices_temp;
        }
    }
    
public:
    GPUOrderBy(int32_t dataElementCount)
    {
        GPUMemory::alloc(&indices1, dataElementCount);
        GPUMemory::alloc(&indices2, dataElementCount);
        GPUMemory::alloc(&keys1, dataElementCount);
        GPUMemory::alloc(&keys2, dataElementCount);

        radix_temp_buf_size_ = 0;
        radix_temp_buf_ = nullptr;
        cub::DeviceRadixSort::SortPairs(radix_temp_buf_, 
                                        radix_temp_buf_size_,
                                        keys1, 
                                        keys2, 
                                        indices1, 
                                        indices2,
                                        dataElementCount);

        GPUMemory::alloc(&radix_temp_buf_, radix_temp_buf_size_);
    }

    ~GPUOrderBy()
    {
        GPUMemory::free(indices1);
        GPUMemory::free(indices2);
        GPUMemory::free(keys1);
        GPUMemory::free(keys2);

        GPUMemory::free(radix_temp_buf_);
    }

    void OrderByAsc(std::vector<int32_t*> &outIndices, std::vector<T*> &inCols, int32_t dataElementCount)
    {
        OrderBy(outIndices, inCols, dataElementCount, true);
    }

    void OrderByDsc(std::vector<int32_t*> &outIndices, std::vector<T*> &inCols, int32_t dataElementCount)
    {
        OrderBy(outIndices, inCols, dataElementCount, false);
    }
    
    void ReOrderByIdx(T* outCol, int32_t* inIndices, T* inCol, int32_t dataElementCount)
    {
        // Reorder a column based on indices
        kernel_reorder_by_idx<<< Context::getInstance().calcGridDim(dataElementCount), 
                                 Context::getInstance().getBlockDim() >>>
                                 (outCol, inIndices, inCol, dataElementCount);
    }
};