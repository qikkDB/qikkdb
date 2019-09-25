#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"
#include "LogicOperations.h"

/// A bitwise relation logic operation kernel
/// <param name="OP">Template parameter for the choice of the logic relation operation</param>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template <typename OP, typename T, typename U, typename V>
__global__ void kernel_logic(T* outCol, U ACol, V BCol, int8_t* nullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (nullBitMask)
        {
            int bitMaskIdx = (i / (sizeof(char) * 8));
            int shiftIdx = (i % (sizeof(char) * 8));
            outCol[i] =
                OP{}.template operator()<T, typename std::remove_pointer<U>::type, typename std::remove_pointer<V>::type>(
                    maybe_deref(ACol, i), maybe_deref(BCol, i)) &&
                !((nullBitMask[bitMaskIdx] >> shiftIdx) & 1);
        }
        else
        {
            outCol[i] =
                OP{}.template operator()<T, typename std::remove_pointer<U>::type, typename std::remove_pointer<V>::type>(
                    maybe_deref(ACol, i), maybe_deref(BCol, i));
        }
    }
}

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
    /// Relational binary logic operation with two columns of numerical values
    /// <param name="OP">Template parameter for the choice of the logic operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    template <typename OP, typename T, typename U>
    static void Logic(int8_t* outMask, T ACol, U BCol, int8_t* nullBitMask, int32_t dataElementCount)
    {
        if (std::is_pointer<T>::value || std::is_pointer<U>::value)
        {
            kernel_logic<OP>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    outMask, ACol, BCol, nullBitMask, dataElementCount);
        }
        else
        {
            kernel_logic<OP>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    outMask, maybe_deref(ACol, 0), maybe_deref(BCol, 0), nullptr, dataElementCount);
        
		}
        CheckCudaError(cudaGetLastError());
    }


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
