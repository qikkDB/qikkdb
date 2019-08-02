#pragma once

#include "../../GpuSqlParser/DispatcherCpu/CpuFilterInterval.h"

/// Functors for parallel binary filtration operations
namespace FilterConditions
{
/// A greater than operator > functor
struct greater
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a > b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        return false; // TODO
    }
};

/// A greater than or equal operator >= functor
struct greaterEqual
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a >= b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        return false; // TODO
    }
};

/// A less than operator < functor
struct less
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a < b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        return false; // TODO
    }
};

/// A less than or equal operator <= functor
struct lessEqual
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a <= b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        return false; // TODO
    }
};

/// An equality operator == functor
struct equal
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::INNER;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a == b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        if (aLength != bLength)
        {
            return false;
        }
        else
        {
            for (int32_t j = 0; j < aLength; j++)
            {
                if (a[j] != b[j])
                {
                    return false;
                }
            }
            return true;
        }
    }
};

/// An unequality operator != functor
struct notEqual
{
    static constexpr CpuFilterInterval interval = CpuFilterInterval::OUTER;
    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a != b;
    }

    __device__ __host__ bool compareStrings(const char* a, int32_t aLength, const char* b, int32_t bLength)
    {
        if (aLength != bLength)
        {
            return true;
        }
        else
        {
            for (int32_t j = 0; j < aLength; j++)
            {
                if (a[j] != b[j])
                {
                    return true;
                }
            }
            return false;
        }
    }
};
} // namespace FilterConditions
