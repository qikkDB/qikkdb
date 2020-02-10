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
template <typename T, typename U>
__global__ void kernel_operator_not(T* outCol, U ACol, int8_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (nullBitMask)
        {
            int bitMaskIdx = (i / (sizeof(char) * 8));
            int shiftIdx = (i % (sizeof(char) * 8));
            outCol[i] = !maybe_deref<typename std::remove_pointer<U>::type>(ACol, i) &&
                        !((nullBitMask[bitMaskIdx] >> shiftIdx) & 1);
        }
        else
        {
            outCol[i] = !maybe_deref<typename std::remove_pointer<U>::type>(ACol, i);
        }
    }
}

/// A class for relational logic operations
class GPULogic
{
public:
    /// Unary NOT operation on a column of data
    /// <param name="outCol">block of the result data</param>
    /// <param name="ACol">block of the input operands</param>
    /// <param name="dataElementCount">the size of the input blocks in elements</param>
    template <typename T, typename U>
    static void Not(T* outCol, U ACol, int8_t* nullBitMask, int32_t dataElementCount)
    {
        Context& context = Context::getInstance();
        kernel_operator_not<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(outCol, ACol, nullBitMask,
                                                                                              dataElementCount);

        // Get last error
        CheckCudaError(cudaGetLastError());
    }
};
