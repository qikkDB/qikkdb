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
#include "cuda_ptr.h"
#include "GPUReconstruct.cuh"

__device__ NativeGeoPoint CastNativeGeoPoint(char* str, int32_t length);

__device__ NativeGeoPoint CastWKTPoint(char* str, int32_t length);


template <typename T>
__device__ int32_t GetNumberOfIntegralDigits(T val)
{
    return (floorf(fabsf(val)) > 3.0f ? static_cast<int32_t>(log10f(floorf(fabsf(val)))) : 0) + 1;
}

template <typename T>
__device__ int32_t GetNumberOfDecimalDigits(T val)
{
    int decimalPart = 0;
    while (fabsf(val) != floorf(fabsf(val)))
    {
        val *= 10;
        decimalPart++;
    }

    return decimalPart;
}

template <typename T>
__device__ T CastDecimal(char* str, int32_t length)
{
    T out = 0;
    int32_t decimalPart = 0;
    int32_t outSign = 1;

    if (*str == '-')
    {
        outSign = -1;
        length--;
        str++;
    }
    else if (*str == '+')
    {
        length--;
        str++;
    }

    while (*str >= '0' && *str <= '9' && length > 0)
    {
        out = out * 10 + (*str - '0');
        length--;
        str++;
    }

    if ((*str == '.' || *str == ',') && length > 0)
    {
        length--;
        str++;
        while (*str >= '0' && *str <= '9' && length > 0)
        {
            out = out * 10 + (*str - '0');
            decimalPart--;
            length--;
            str++;
        }
    }

    if ((*str == 'e' || *str == 'E') && length > 0)
    {
        length--;
        str++;
        int32_t sign = 1;
        int32_t afterEPart = 0;

        if (*str == '-')
        {
            sign = -1;
            length--;
            str++;
        }

        else if (*str == '+')
        {
            length--;
            str++;
        }

        while (*str >= '0' && *str <= '9' && length > 0)
        {
            afterEPart = afterEPart * 10 + (*str - '0');
            length--;
            str++;
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

template <typename IN>
__global__ void kernel_predict_numeric_string_lengths(int32_t* outStringLengths, IN inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int32_t decimalDigits = GetNumberOfDecimalDigits(maybe_deref(inCol, i));
        outStringLengths[i] = ((maybe_deref(inCol, i) < 0) ? 1 : 0) +
                              GetNumberOfIntegralDigits(maybe_deref(inCol, i)) +
                              ((decimalDigits > 0) ? (decimalDigits + 1) : 0);
    }
}

template <typename IN>
__device__ void NumericToString(char* allChars, int64_t startIndex, IN number, int32_t fixedIntegralDigits = 0)
{
    // Append sign
    if (number < 0)
    {
        allChars[startIndex++] = '-';
        // (note that there is no addres move because we will count with negative sign later)
    }

    // Append integer part
    int64_t integerPart = static_cast<int64_t>(floorf(fabsf(number)));

    const int32_t integralDigits = GetNumberOfIntegralDigits(number);

    fixedIntegralDigits = max(fixedIntegralDigits, integralDigits);

    startIndex += fixedIntegralDigits;
    do
    {
        allChars[--startIndex] = ('0' + (integerPart % 10));
        integerPart /= 10;
    } while (integerPart > 0);

    for (int32_t i = 0; i < fixedIntegralDigits - integralDigits; i++)
    {
        allChars[--startIndex] = '0';
    }

    startIndex += fixedIntegralDigits;

    // Append decimal part
    int32_t decimalDigits = GetNumberOfDecimalDigits(number);

    if (decimalDigits > 0)
    {
        int32_t decimalPart =
            static_cast<int32_t>(roundf(fmodf(fabsf(number), 1.0f) * powf(10.0f, decimalDigits)));
        allChars[startIndex++] = '.';
        startIndex += decimalDigits;
        for (int32_t i = 0; i < decimalDigits; i++) // Fixed decimal places
        {
            allChars[--startIndex] = ('0' + (decimalPart % 10));
            decimalPart /= 10;
        }
        startIndex += decimalDigits;
    }
}

namespace CastOperations
{

struct FromString
{
    template <typename OUT>
    __device__ OUT operator()(char* str, int32_t length) const;
};

struct ToString
{
    typedef std::string RetType;
};

template <>
__device__ int32_t FromString::operator()<int32_t>(char* str, int32_t length) const;

template <>
__device__ int64_t FromString::operator()<int64_t>(char* str, int32_t length) const;

template <>
__device__ float FromString::operator()<float>(char* str, int32_t length) const;

template <>
__device__ double FromString::operator()<double>(char* str, int32_t length) const;

template <>
__device__ NativeGeoPoint FromString::operator()<NativeGeoPoint>(char* str, int32_t length) const;

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

template <typename IN>
__global__ void kernel_cast_numeric_to_string(GPUMemory::GPUString outCol, IN inCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int64_t stringIndex = GetStringIndex(outCol.stringIndices, i);
        NumericToString(outCol.allChars, stringIndex, maybe_deref(inCol, i));
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

    template <typename IN>
    static void CastNumericToString(GPUMemory::GPUString& outCol, IN inCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<typename std::remove_pointer<IN>::type>::value,
                      "InCol must be arithmetic data type");

        cuda_ptr<int32_t> stringLengths(dataElementCount);
        kernel_predict_numeric_string_lengths<<<Context::getInstance().calcGridDim(dataElementCount),
                                                Context::getInstance().getBlockDim()>>>(
            stringLengths.get(), inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUMemory::alloc(&(outCol.stringIndices), dataElementCount);
        GPUReconstruct::PrefixSum(outCol.stringIndices, stringLengths.get(), dataElementCount);


        int64_t totalCharCount;
        GPUMemory::copyDeviceToHost(&totalCharCount, outCol.stringIndices + dataElementCount - 1, 1);
        GPUMemory::alloc(&(outCol.allChars), totalCharCount);

        kernel_cast_numeric_to_string<<<Context::getInstance().calcGridDim(dataElementCount),
                                        Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    template <typename OUT>
    static void CastString(OUT* outCol, GPUMemory::GPUString inCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<OUT>::value || std::is_same<OUT, NativeGeoPoint>::value,
                      "OutCol must be arithmetic or point data type");

        kernel_cast_string<<<Context::getInstance().calcGridDim(dataElementCount),
                             Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);

        CheckCudaError(cudaGetLastError());
    }
};