#pragma once

#include "cuda_ptr.h"

__host__ __device__ constexpr RADIX_BUCKET_COUNT = 65536;

class GPUOrderBy {
private:
    cuda_ptr<int32_t> radix_bucket_histo_;
    cuda_ptr<int32_t> radix_bucket_prefix_sum_;
    
public:
    GPUOrderBy() :
        radix_bucket_histo_(RADIX_BUCKET_COUNT),
        radix_bucket_prefix_sum_RADIX_BUCKET_COUNT)
    {
        // Constructor empty
    }

    template<typename T>
    void OrderBy(int32_t* outColIndices, std::vector<T*> &inCols, int32_t dataElementCount)
    {

    }
    
    template<typename T>
    void ReorderByIdx(T* outCol, int32_t* inColIndices, T* inCol, int32_t dataElementCount)
    {
        
    }
};