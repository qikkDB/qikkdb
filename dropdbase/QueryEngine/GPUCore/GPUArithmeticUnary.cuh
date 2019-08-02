#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "ArithmeticUnaryOperations.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic unary operation with column and column
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template <typename OP, typename T, typename U>
__global__ void kernel_arithmetic_unary(T* output, U ACol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] = OP{}.template operator()<T, typename std::remove_pointer<U>::type>(maybe_deref(ACol, i));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for unary arithmetic functions
class GPUArithmeticUnary
{
public:
    /// Arithmetic unary operation with values from column
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    template <typename OP, typename T, typename U>
    static void col(T* output, U* ACol, int32_t dataElementCount)
    {
        kernel_arithmetic_unary<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, ACol, dataElementCount);
    }

    /// Arithmetic unary operation with constant
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="AConst">operand (constant)</param>
    /// <param name="dataElementCount">data element count of the output buffer (how many times copy result)</param>
    template <typename OP, typename T, typename U>
    static void cnst(T* output, U AConst, int32_t dataElementCount)
    {
        kernel_arithmetic_unary<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, AConst, dataElementCount);
    }
};
