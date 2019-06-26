#pragma once

// Fill the index buffers with default indices
__global__ kernel_fill_indices(int32_t* indices, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        indices[i] = i;
	}
}

// Reorder the keys and the values
__host__ __device__ constexpr uint64_t RADIX_BUCKET_COUNT = 65536;
__host__ __device__ constexpr uint64_t RADIX_MASK = 0xFFFF;
__host__ __device__ constexpr uint64_t RADIX_MASK_BIT_WIDTH = 16;

// Calculate the histogram of occurrances of keys (like in counting sort)
template<typename T>
__global__ kernel_radix_histo(int32_t* radix_histo, 
                              T* keys,
                              int32_t radix_pass,
                              bool is_signed_pass,
                              int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        int32_t radixHistoIdx = (keys[i] & (RADIX_MASK << (RADIX_MASK_BIT_WIDTH * radix_pass))) >> 
                                (RADIX_MASK_BIT_WIDTH * radix_pass);
        atomicAdd(&radix_histo[radixHistoIdx], 1);
	}
}

// Reorder the indices and the keys based on the radix and data type
template<typename T>
__global__ kernel_radix_sort(int32_t* indicesOut, 
                             T* keysOut,
                             int32_t* indicesIn,
                             T* keysIn,
                             int32_t* radix_pref_sum,
                             int32_t radix_pass,
                             bool is_signed_pass,
                             int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
        
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

    // Radix key histogram and prefix sum buffer
    int32_t* radix_histo_;
    int32_t* radix_pref_sum_;

    // Prefix sum helper buffers
    size_t radix_pref_sum_temp_buf_size_;
    int8_t* radix_pref_sum_temp_buf_;

    // One iteration of the radix sort algorithm put in one function
    void RadixPass(int32_t* indicesOut, 
                   T* keysOut,
                   int32_t* indicesIn,
                   T* keysIn,
                   int32_t radix_pass,
                   bool is_signed_pass,
                   int32_t dataElementCount) {
        // Zero the histogram
        GPUMemory::fillArray(radix_histo_, 0, RADIX_BUCKET_COUNT);

        // Calcualte the radix histogram
        kernel_radix_histo<<< Context::getInstance().calcGridDim(dataElementCount), 
                            Context::getInstance().getBlockDim() >>>
                            (radix_histo_, 
                            keysIn, 
                            radix_pass, 
                            is_signed_pass, 
                            dataElementCount);

        // Calcualte the histogram inclusive prefix sum
        cub::DeviceScan::InclusiveSum(radix_pref_sum_temp_buf_, 
                                    radix_pref_sum_temp_buf_size_, 
                                    radix_histo_, 
                                    radix_pref_sum_, 
                                    dataElementCount);

        // Perform the radix sort on the keys, order the indices with the keys too
        kernel_radix_sort<<< Context::getInstance().calcGridDim(dataElementCount), 
                            Context::getInstance().getBlockDim() >>>
                            (indicesOut, 
                            keysOut,
                            indicesIn,
                            keysIn,
                            radix_pref_sum_temp_buf_, 
                            radix_pass, 
                            is_signed_pass, 
                            dataElementCount);
    }
    
public:
    GPUOrderBy(int32_t dataElementCount)
    {
        GPUMemoryAlloc(&indices1, dataElementCount);
        GPUMemoryAlloc(&indices2, dataElementCount);
        GPUMemoryAlloc(&keys1, dataElementCount);
        GPUMemoryAlloc(&keys2, dataElementCount);

        GPUMemoryAlloc(&radix_histo_, RADIX_BUCKET_COUNT);
        GPUMemoryAlloc(&radix_pref_sum_, RADIX_BUCKET_COUNT);

        radix_pref_sum_temp_buf_size_ = 0;
        cub::DeviceScan::InclusiveSum(nullptr, 
                                      radix_pref_sum_temp_buf_size_, 
                                      radix_histo_, 
                                      radix_pref_sum_, 
                                      dataElementCount);
        GPUMemoryAlloc(&radix_pref_sum_temp_buf_, radix_pref_sum_temp_buf_size_);
    }

    ~GPUOrderBy()
    {
        GPUMemoryFree(indices1);
        GPUMemoryFree(indices2);
        GPUMemoryFree(keys1);
        GPUMemoryFree(keys2);

        GPUMemoryFree(radix_histo_);
        GPUMemoryFree(radix_pref_sum_);
        GPUMemoryFree(radix_pref_sum_temp_buf_);
    }

    void OrderBy(int32_t* outColIndices, std::vector<T*> &inCols, int32_t dataElementCount)
    {
        // FInitialize the index buffer
        kernel_fill_indices<<< Context::getInstance().calcGridDim(dataElementCount), 
                               Context::getInstance().getBlockDim() >>>
                               (indices1, dataElementCount);

        // Iterate trough all the columns and sort them with radix sort
        // Handle the columns as if they were a big binary number from right to left
        for(int32_t i = 0; i < inCols.size()l i++)
        {
            // Copy the keys to the first key buffer
            GPUMemory::copyDeviceToDevice(keys1, inCols[i], dataElementCount)

            // Passes for different data types
            if(std::is_same<T, int32_t>::value || std::is_same<T, float>::value )
            {
                // Signed 32 bit integers or 32 bit float values
                // First pass - unsigned lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 0, false, dataElementCount);

                // Second pass - signed upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 1, true, dataElementCount);
            } 
            else if(std::is_same<T, uint32_t>::value)
            {
                // Unsigned 32 bit integers
                // First pass - unsigned lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 0, false, dataElementCount);

                // Second pass - signed upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 1, false, dataElementCount);
            }
            else if(std::is_same<T, int64_t>::value || std::is_same<T, double>::value )
            {
                // Signed 64 bit integers or 64 bit double values
                // First pass - unsigned lower lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 0, false, dataElementCount);

                // Second pass - signed lower upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 1, false, dataElementCount);

                // Third pass - unsigned upper lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 2, false, dataElementCount);

                // Fourth pass - signed upper upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 3, true, dataElementCount);
            }
            else if(std::is_same<T, int64_t>::value)
            {
                // Unsigned 64 bit integers
                // First pass - unsigned lower lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 0, false, dataElementCount);

                // Second pass - signed lower upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 1, false, dataElementCount);

                // Third pass - unsigned upper lower 16 bits
                RadixPass(indices2, keys2, indices1, keys1, 2, false, dataElementCount);

                // Fourth pass - signed upper upper 16 bits
                RadixPass(indices1, keys1, indices2, keys2, 3, false, dataElementCount);
            }
        }
    }
    
    void ReOrderByIdx(T* outCol, int32_t* inColIndices, T* inCol, int32_t dataElementCount)
    {

    }
};