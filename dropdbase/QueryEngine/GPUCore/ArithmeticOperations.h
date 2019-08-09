#pragma once
#include <cmath>

/// Namespace for arithmetic operation generic functors
namespace ArithmeticOperations
{
struct addNoCheck
{
    static constexpr bool isMonotonous = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return a + b;
    }
};

struct subNoCheck
{
    static constexpr bool isMonotonous = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return a - b;
    }
};

struct mulNoCheck
{
    static constexpr bool isMonotonous = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return a * b;
    }
};

struct divNoCheck
{
    static constexpr bool isMonotonous = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return a / b;
    }
};

struct modNoCheck
{
    static constexpr bool isMonotonous = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        // modulo is not defined for floating point type
        static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
                      "None of the input columns of operation modulo cannot be floating point "
                      "type!");

        return a % b;
    }
};

struct bitwiseAndNoCheck
{
    static constexpr bool isMonotonous = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a & b;
    }
};

struct bitwiseOrNoCheck
{
    static constexpr bool isMonotonous = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a | b;
    }
};

struct bitwiseXorNoCheck
{
    static constexpr bool isMonotonous = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a ^ b;
    }
};

struct bitwiseLeftShiftNoCheck
{
    static constexpr bool isMonotonous = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a << b;
    }
};

struct bitwiseRightShiftNoCheck
{
    static constexpr bool isMonotonous = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a >> b;
    }
};

struct logarithmNoCheck
{
    static constexpr bool isMonotonous = true;
    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return logf(a) / logf(b);
    }
};

struct arctangent2NoCheck
{
    static constexpr bool isMonotonous = true;
    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return atan2f(a, b);
    }
};

struct powerNoCheck
{
    static constexpr bool isMonotonous = false;
    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return powf(a, b);
    }
};

struct rootNoCheck
{
    static constexpr bool isMonotonous = false;
    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b) const
    {
        return powf(a, 1.0f / b);
    }
};
} // namespace ArithmeticOperations