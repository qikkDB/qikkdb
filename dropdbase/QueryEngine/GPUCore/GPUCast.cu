#include "GPUCast.cuh"

template <>
__device__ int32_t CastOperations::FromString::operator()<int32_t>(char* str, int32_t length) const
{
    return CastIntegral(str, length);
}

template <>
__device__ float CastOperations::FromString::operator()<float>(char* str, int32_t length) const
{
    return CastDecimal<float>(str, length);
}

template <>
__device__ double CastOperations::FromString::operator()<double>(char* str, int32_t length) const
{
    return CastDecimal<double>(str, length);
}

__device__ int32_t CastIntegral(char* str, int32_t length, int32_t base)
{
    int32_t out = 0;
    int32_t order = 1;
    int32_t sign = 1;
    int32_t numBoundary = 0;

    if (str[0] == '-')
    {
        sign = -1;
        numBoundary = 1;
    }

    for (int32_t i = length - 1; i >= numBoundary; i--)
    {
        out += (str[i] - '0') * order;
        order *= base;
    }
    return sign * out;
}