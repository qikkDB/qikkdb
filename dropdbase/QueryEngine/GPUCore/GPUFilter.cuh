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
#include "GPUArithmetic.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for comparing values
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template <typename OP, typename T, typename U>
__global__ void kernel_filter(int8_t* outMask, T ACol, U BCol, int8_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (nullBitMask)
        {
            int bitMaskIdx = (i / (sizeof(char) * 8));
            int shiftIdx = (i % (sizeof(char) * 8));
            outMask[i] =
                OP{}.template operator()<typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type>(
                    maybe_deref(ACol, i), maybe_deref(BCol, i)) &&
                !((nullBitMask[bitMaskIdx] >> shiftIdx) & 1);
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
__global__ void kernel_filter_string(int8_t* outMask,
                                     GPUMemory::GPUString inputA,
                                     bool isACol,
                                     GPUMemory::GPUString inputB,
                                     bool isBCol,
                                     int8_t* nullBitMask,
                                     int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int bitMaskIdx;
        int shiftIdx;
        if (nullBitMask != nullptr)
        {
            bitMaskIdx = (i / (sizeof(char) * 8));
            shiftIdx = (i % (sizeof(char) * 8));
        }
        const int32_t aI = isACol ? i : 0;
        const int64_t aIndex = GetStringIndex(inputA.stringIndices, aI);
        const int32_t aLength = static_cast<int32_t>(inputA.stringIndices[aI] - aIndex);
        const int32_t bI = isBCol ? i : 0;
        const int64_t bIndex = GetStringIndex(inputB.stringIndices, bI);
        const int32_t bLength = static_cast<int32_t>(inputB.stringIndices[bI] - bIndex);
        if (nullBitMask != nullptr)
        {
            outMask[i] = OP{}.compareStrings(inputA.allChars + aIndex, aLength, inputB.allChars + bIndex, bLength) &&
                         !((nullBitMask[bitMaskIdx] >> shiftIdx) & 1);
        }
        else
        {
            outMask[i] = OP{}.compareStrings(inputA.allChars + aIndex, aLength, inputB.allChars + bIndex, bLength);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// A class for column-wise logic operation for data filtering based on logic conditions
class GPUFilter
{
public:
    /// Filtration operation with two columns of binary masks
    /// <param name="OP">Template parameter for the choice of the filtration operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    template <typename OP, typename T, typename U>
    static void Filter(int8_t* outMask, T ACol, U BCol, int8_t* nullBitMask, int32_t dataElementCount)
    {
        if (std::is_pointer<T>::value || std::is_pointer<U>::value)
        {
            kernel_filter<OP>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    outMask, ACol, BCol, nullBitMask, dataElementCount);
        }
        else
        {
            GPUMemory::memset(outMask, OP{}(maybe_deref(ACol, 0), maybe_deref(BCol, 0)), dataElementCount);
        }
        CheckCudaError(cudaGetLastError());
    }


    /// Filtration operation between two strings (column-column)
    template <typename OP>
    static void FilterString(int8_t* outMask,
                             GPUMemory::GPUString ACol,
                             bool isACol,
                             GPUMemory::GPUString BCol,
                             bool isBCol,
                             int8_t* nullBitMask,
                             int32_t dataElementCount)
    {
        kernel_filter_string<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                outMask, ACol, isACol, BCol, isBCol, nullBitMask, dataElementCount);
        CheckCudaError(cudaGetLastError());

        if (!isACol && !isBCol)
        {
            // Expand mask - copy the one result to whole mask
            int8_t numberFromMask;
            GPUMemory::copyDeviceToHost(&numberFromMask, outMask, 1);
            GPUMemory::memset(outMask, numberFromMask, dataElementCount);
        }
    }
};
