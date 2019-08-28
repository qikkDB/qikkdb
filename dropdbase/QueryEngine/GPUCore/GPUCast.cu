#include "GPUCast.cuh"

template <>
__device__ int32_t CastOperations::FromString::operator()<int32_t>(char* str, int32_t length) const
{
    return CastDecimal<int32_t>(str, length);
}

template <>
__device__ int64_t CastOperations::FromString::operator()<int64_t>(char* str, int32_t length) const
{
    return CastDecimal<int64_t>(str, length);
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