#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "DateOperations.h"

/// Kernel for extracting date or time variable (e.g. days, hours)
/// from datetime column or constant
/// <param name="output">block of the result data</param>
/// <param name="dateTimeCol">input timestamp (column or constant)</param>
/// <param name="dataElementCount">the count of elements in the input block
/// (or of output block if input is constant)</param>
template <typename OP, typename T>
__global__ void kernel_extract(int32_t* output, T dateTimeCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] = OP{}.operator()(maybe_deref(dateTimeCol, i));
    }
}

/// GPUDate class is for extracting (conversion) variables (e.g. days, hours)
/// from datetime column or constant
class GPUDate
{
public:
    /// Extract values (one value - year/month/dat/hour/minute/second - per row) from a datetime column
    /// <param name="output">Output GPU buffer for result (int32_t)</param>
    /// <param name="dateTimeCol">Input GPU buffer - datetime (int64_t)</param>
    /// <param name="dataElementCount">Row count (e.g. size of block to process)</param>
    template <typename OP, typename T>
    static void Extract(int32_t* output, T dateTimeCol, int32_t dataElementCount)
    {
        static_assert(std::is_same<typename std::remove_pointer<T>::type, int64_t>::value,
                      "DateTime can only be extracted from int64 columns");
        kernel_extract<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, dateTimeCol, dataElementCount);
        cudaDeviceSynchronize();
        CheckCudaError(cudaGetLastError());
    }
};
