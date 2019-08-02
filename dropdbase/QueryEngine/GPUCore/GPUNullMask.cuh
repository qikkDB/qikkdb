#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"

namespace NullMaskOperations
{
struct isNull
{
    __device__ int8_t operator()(int8_t a, int32_t bit)
    {
        return (a >> bit) & 1U;
    }
};

struct isNotNull
{
    __device__ int8_t operator()(int8_t a, int32_t bit)
    {
        return ((~a) >> bit) & 1U;
    }
};
} // namespace NullMaskOperations

template <typename OP>
__global__ void kernel_null_mask(int8_t* output, int8_t* AColNullMask, int32_t maskByteSize, int32_t outputSize)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maskByteSize; i += stride)
    {
        int8_t maskElement = AColNullMask[i];
        for (int32_t bit = 0; bit < 8; bit++)
        {
            int32_t outputIdx = 8 * i + bit;
            if (outputIdx < outputSize)
            {
                output[outputIdx] = OP{}(maskElement, bit);
            }
        }
    }
}


/// Class for unary arithmetic functions
class GPUNullMask
{
public:
    template <typename OP>
    static void Col(int8_t* output, int8_t* AColMask, int32_t maskByteSize, int32_t outputSize)
    {
        kernel_null_mask<OP>
            <<<Context::getInstance().calcGridDim(maskByteSize), Context::getInstance().getBlockDim()>>>(
                output, AColMask, maskByteSize, outputSize);
    }
};