#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUFilterConditions.cuh"
#include "MaybeDeref.cuh"
#include "GPUStringUnary.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for comparing values
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template <typename OP, typename T, typename U>
__global__ void kernel_filter(int64_t* outMask, T ACol, U BCol, int64_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (nullBitMask)
        {
            outMask[i] =
                OP{}.template operator()<typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type>(
                    maybe_deref(ACol, i), maybe_deref(BCol, i)) &&
                !(NullValues::GetConcreteBitFromBitmask(nullBitMask, i));
        }
        else
        {
            outMask[i] =
                OP{}.template operator()<typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type>(
                    maybe_deref(ACol, i), maybe_deref(BCol, i));
        }
    }
}

/// Kernel for string comparison (equality, ...)
template <typename OP>
__global__ void kernel_filter_string(int64_t* outMask,
                                     GPUMemory::GPUString inputA,
                                     bool isACol,
                                     GPUMemory::GPUString inputB,
                                     bool isBCol,
                                     int64_t* nullBitMask,
                                     int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int32_t aI = isACol ? i : 0;
        const int64_t aIndex = GetStringIndex(inputA.stringIndices, aI);
        const int32_t aLength = static_cast<int32_t>(inputA.stringIndices[aI] - aIndex);
        const int32_t bI = isBCol ? i : 0;
        const int64_t bIndex = GetStringIndex(inputB.stringIndices, bI);
        const int32_t bLength = static_cast<int32_t>(inputB.stringIndices[bI] - bIndex);
        if (nullBitMask != nullptr)
        {
            outMask[i] = OP{}.compareStrings(inputA.allChars + aIndex, aLength, inputB.allChars + bIndex, bLength) &&
                !(NullValues::GetConcreteBitFromBitmask(nullBitMask, i));
        }
        else
        {
            outMask[i] = OP{}.compareStrings(inputA.allChars + aIndex, aLength, inputB.allChars + bIndex, bLength);
        }
    }
}