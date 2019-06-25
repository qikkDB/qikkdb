#pragma once

#include "GPUMemory.cuh"
#include "cuda_ptr.h"

#include "../../../cub/cub.cuh"

// Constants for radix sort
__host__ __device__ constexpr RADIX_MASK = 0xFFFF;
__host__ __device__ constexpr RADIX_BUCKET_COUNT = 65536;

// Initialize the index buffer for reshuffling based on order by operation during the radix sort
__global__ void kernel_init_idx_buffer(int32_t* radix_index_buffer, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        radix_index_buffer[i] = i;
	}
}

// Perform the index rotation - by 16 bits
template<typename T>
__global__ 
void kernel_order_by(int32_t* radix_bucket_histo, T* inCol, int32_t dataElementCount, uint32_t radixMaskShift)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        uint32_t radixShiftVal = static_cast<uint32_t>(inCol[i]);

        int32_t radixIndex = (radixShiftVal & (RADIX_MASK << radixMaskShift)) >> radixMaskShift;
        atomicAdd(&radix_bucket_histo[radixIndex], 1);
	}
}

// Reorder the rotated indices based on the prefix sum
template<typename T>
__global__
void kernel_reorder_by_idx()
{

}

template<typename T>
class GPUOrderBy {
private:
    int32_t* radix_bucket_histo_;

    size_t radix_bucket_prefix_sum_size_;
    int8_t* radix_bucket_prefix_sum_temp_;
    int32_t* radix_bucket_prefix_sum_;

    int32_t* radix_index_buffer_1_;
    int32_t* radix_index_buffer_2_;

    T* radix_buffer_1_;
    T* radix_buffer_2_;
    
public:
    GPUOrderBy(int32_t dataElementCount)
    {
        GPUMemory::alloc(&radix_bucket_histo_, RADIX_BUCKET_COUNT);
        GPUMemory::alloc(&radix_bucket_prefix_sum_, RADIX_BUCKET_COUNT);

        // Preallocate a helper buffer for the prefix sum
        radix_bucket_prefix_sum_size_ = 0;
		cub::DeviceScan::InclusiveSum(nullptr, radix_bucket_prefix_sum_size_, radix_bucket_histo_, radix_bucket_prefix_sum_, RADIX_BUCKET_COUNT);
		GPUMemory::alloc(&radix_bucket_prefix_sum_temp_, radix_bucket_prefix_sum_size_);
        
        GPUMemory::alloc(&radix_index_buffer_1_, dataElementCount);
        GPUMemory::alloc(&radix_index_buffer_2_, dataElementCount);
        GPUMemory::alloc(&radix_buffer_1_, dataElementCount);
        GPUMemory::alloc(&radix_buffer_2_, dataElementCount);

        // Fill the first index buffer with the default indices
        kernel_init_idx_buffer<<<Context::getInstance().calcGridDim(dataElementCount), 
                                 Context::getInstance().getBlockDim()>>>
                                 (radix_index_buffer_, dataElementCount);
    }

    ~GPUOrderBy()
    {
        GPUMemory::free(radix_bucket_histo_);
        GPUMemory::free(radix_bucket_prefix_sum_);
        
        GPUMemory::free(radix_bucket_prefix_sum_temp_);

        GPUMemory::free(radix_index_buffer_1_);
        GPUMemory::free(radix_index_buffer_2_);
        GPUMemory::free(radix_buffer_1_);
        GPUMemory::free(radix_buffer_2_);
    }

    void OrderBy(int32_t* outColIndices, std::vector<T*> &inCols, int32_t dataElementCount)
    {
        for(int32_t i = inCols.size() - 1; i >= 0; i--)
        {
            // Zero the histogram and the radix buffers
            GPUMemory::fillArray(radix_bucket_histo_, 0, dataElementCount);
            GPUMemory::fillArray(radix_buffer_1_, 0, dataElementCount);
            GPUMemory::fillArray(radix_buffer_2_, 0, dataElementCount);

            // Count the radix occurrances then reorder the indices 
            kernel_order_by<<<Context::getInstance().calcGridDim(dataElementCount), 
                              Context::getInstance().getBlockDim()>>>
                              (radix_bucket_histo_, &inCols[i], dataElementCount, 0);

            // Calculate the prefix sum for the histogram
            cub::DeviceScan::InclusiveSum(radix_bucket_prefix_sum_temp_, radix_bucket_prefix_sum_size_, radix_bucket_histo_, radix_bucket_prefix_sum_, RADIX_BUCKET_COUNT);
            
            // Reorder the input keys partially by the radix
            

        }
    }
    
    
    void ReorderByIdx(T* outCol, int32_t* inColIndices, T* inCol, int32_t dataElementCount)
    {

    }
};