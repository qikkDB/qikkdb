#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"

template <typename OUT, typename IN>
__global__ void kernel_cast_numeric(OUT* outCol, IN inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        outCol[i] = static_cast<OUT>(maybe_deref(inCol, i));
    }
}

class GPUCast
{
public:
    template <typename OUT, typename IN>
    static void CastNumericCol(OUT* outCol, IN* inCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<IN>::value, "InCol must be arithmetic data type");
        static_assert(std::is_arithmetic<OUT>::value, "OutCol must be arithmetic data type");

        kernel_cast_numeric<<<Context::getInstance().calcGridDim(dataElementCount),
                              Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    template <typename OUT, typename IN>
    static void CastNumericConst(OUT* outCol, IN inConst, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<IN>::value, "InCol must be arithmetic data type");
        static_assert(std::is_arithmetic<OUT>::value, "OutCol must be arithmetic data type");

        kernel_cast_numeric<<<Context::getInstance().calcGridDim(dataElementCount),
                              Context::getInstance().getBlockDim()>>>(outCol, inConst, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }
};