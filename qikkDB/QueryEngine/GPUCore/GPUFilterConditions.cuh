#pragma once

#include "../../GpuSqlParser/DispatcherCpu/CpuFilterInterval.h"

/// Functors for parallel binary filtration operations
namespace FilterConditions
{
/// A greater than operator > functor
struct greater
{
    typedef int8_t RetType;
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
    typedef int8_t RetType;
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
    typedef int8_t RetType;
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
    typedef int8_t RetType;
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
    typedef int8_t RetType;
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
    typedef int8_t RetType;
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

/// A logical binary AND operation
struct logicalAnd
{
    typedef int8_t RetType;
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;

    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a && b;
    }
};

/// A logical binary OR operation
struct logicalOr
{
    typedef int8_t RetType;
    static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;

    template <typename T, typename U>
    __device__ __host__ int8_t operator()(T a, U b) const
    {
        return a || b;
    }
};

struct logicalNot
{
    static constexpr bool isMonotonous = false;
    typedef int8_t RetType;
};

template <typename OP>
constexpr bool isFilterOp = std::is_same<OP, FilterConditions::greater>::value ||
                            std::is_same<OP, FilterConditions::greaterEqual>::value ||
                            std::is_same<OP, FilterConditions::less>::value ||
                            std::is_same<OP, FilterConditions::lessEqual>::value ||
                            std::is_same<OP, FilterConditions::equal>::value ||
                            std::is_same<OP, FilterConditions::notEqual>::value ||
                            std::is_same<OP, FilterConditions::logicalAnd>::value ||
                            std::is_same<OP, FilterConditions::logicalOr>::value;

} // namespace FilterConditions
