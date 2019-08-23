#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"
#include "GPUStringUnary.cuh"

__device__ int32_t CastIntegral(char* str, int32_t length, int32_t base = 10);

template <typename T>
__device__ T CastDecimal(char* str, int32_t length)
{
    T out = 0;
    int32_t decimalPart = 0;
    int32_t outSign = 1;

    int c;

    if (*str++ == '-')
    {
        outSign = -1;
        length--;
    }
    else
    {
        str--;
    }

    while (((c = *str++) >= '0' && c <= '9') && length > 0)
    {
        out = out * 10 + (c - '0');
        length--;
    }

    if ((c == '.' || c == ',') && length > 0)
    {
        length--;
        while (((c = *str++) >= '0' && c <= '9') && length > 0)
        {
            out = out * 10 + (c - '0');
            decimalPart--;
            length--;
        }
    }

    else if ((c == 'e' || c == 'E') && length > 0)
    {
        length--;
        int32_t sign = 1;
        int32_t afterEPart = 0;

        c = *str++;
        if (c == '-')
        {
            sign = -1;
            length--;
        }

        else if (c == '+')
        {
            length--;
        }

        while (((c == *str++) >= '0' && c <= '9') && length > 0)
        {
            afterEPart = afterEPart * 10 + (c - '0');
            length--;
        }

        decimalPart += afterEPart * sign;
    }

    while (decimalPart > 0)
    {
        out *= 10;
        decimalPart--;
    }

    while (decimalPart < 0)
    {
        out *= 0.1;
        decimalPart++;
    }

    return out * outSign;
}

namespace CastOperations
{

struct FromString
{
    template <typename OUT>
    __device__ OUT operator()(char* str, int32_t length) const;
};

template <>
__device__ int32_t FromString::operator()<int32_t>(char* str, int32_t length) const;

template <>
__device__ float FromString::operator()<float>(char* str, int32_t length) const;

template <>
__device__ double FromString::operator()<double>(char* str, int32_t length) const;

} // namespace CastOperations

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

template <typename OUT>
__global__ void kernel_cast_string(OUT* outCol, GPUMemory::GPUString inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int64_t strIdx = GetStringIndex(inCol.stringIndices, i);
        const int32_t strLength = GetStringLength(inCol.stringIndices, i);

        outCol[i] = CastOperations::FromString{}.template operator()<OUT>(inCol.allChars + strIdx, strLength);
    }
}

class GPUCast
{
public:
    template <typename OUT, typename IN>
    static void CastNumeric(OUT* outCol, IN inCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<typename std::remove_pointer<IN>::type>::value,
                      "InCol must be arithmetic data type");
        static_assert(std::is_arithmetic<OUT>::value, "OutCol must be arithmetic data type");

        kernel_cast_numeric<<<Context::getInstance().calcGridDim(dataElementCount),
                              Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    template <typename OUT>
    static void CastString(OUT* outCol, GPUMemory::GPUString inCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<OUT>::value, "OutCol must be arithmetic data type");

        kernel_cast_string<<<Context::getInstance().calcGridDim(dataElementCount),
                             Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);

        CheckCudaError(cudaGetLastError());
    }
};
