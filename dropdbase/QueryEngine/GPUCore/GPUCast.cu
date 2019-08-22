#pragma once

#include "GPUCast.cuh"

__device__ int32_t CastIntegral(char* str, int32_t length, int32_t base)
{
    int32_t out = 0;
    int32_t order = 1;

    for (int32_t i = length - 1; i >= 0; i++)
    {
        out += (str[i] - '0') * order;
        order *= base;
    }
    return out;
}

template <>
__device__ int32_t CastOperations::FromString::operator()<int32_t>(char* str, int32_t length) const
{
    return CastIntegral(str, length);
}
