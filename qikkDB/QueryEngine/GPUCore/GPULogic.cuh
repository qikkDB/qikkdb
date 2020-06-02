#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"

/// An unary NOT operation kernel
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template <typename T>
__global__ void kernel_operator_not(int8_t* outCol, T ACol, nullmask_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (nullBitMask)
        {
            outCol[i] = !maybe_deref<typename std::remove_pointer<T>::type>(ACol, i) &&
                        !(NullValues::GetConcreteBitFromBitmask(nullBitMask, i));
        }
        else
        {
            outCol[i] = !maybe_deref<typename std::remove_pointer<T>::type>(ACol, i);
        }
    }
}