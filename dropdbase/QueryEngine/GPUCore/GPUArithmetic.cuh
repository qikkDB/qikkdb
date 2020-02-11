#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "../NullConstants.cuh"
#include "../../MathConstants.h"
#include "ArithmeticOperations.h"
#include "GPUStringBinary.cuh"

namespace ArithmeticOperations
{

/// Arithmetic operation add
struct add
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
        {
            // Check for overflow
            if (((b > V{0}) && (a > (max - b))) || ((b < V{0}) && (a < (min - b))))
            {
                atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                return GetNullConstant<T>();
            }
        }
        return a + b;
    }
};

/// Arithmetic operation subtraction
struct sub
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
        {
            // Check for overflow
            if (((b > V{0}) && (a < (min + b))) || ((b < V{0}) && (a > (max + b))))
            {
                atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                return GetNullConstant<T>();
            }
        }
        return a - b;
    }
};

/// Arithmetic operation multiply
struct mul
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
        {
            // Check for overflow
            if (a > U{0})
            {
                if (b > V{0})
                {
                    if (a > (max / b))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<T>();
                    }
                }
                else
                {
                    if (b < (min / a))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<T>();
                    }
                }
            }
            else
            {
                if (b > V{0})
                {
                    if (a < (min / b))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<T>();
                    }
                }
                else
                {
                    if ((a != U{0}) && (b < (max / a)))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<T>();
                    }
                }
            }
        }
        return a * b;
    }
};

/// Arithmetic operation divide
struct div
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        if (b == V{0})
        {
            atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
            return GetNullConstant<T>();
        }
        else
        {
            return a / b;
        }
    }
};

/// Arithmetic operation modulo
struct mod
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        // modulo is not defined for floating point type
        static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
                      "None of the input columns of operation modulo cannot be floating point "
                      "type!");

        // Check for zero division
        if (b == V{0})
        {
            atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
            return GetNullConstant<T>();
        }

        return a % b;
    }
};

/// Bitwise operation and
struct bitwiseAnd
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
    {
        return a & b;
    }
};

/// Bitwise operation or
struct bitwiseOr
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
    {
        return a | b;
    }
};

/// Bitwise operation xor
struct bitwiseXor
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
    {
        return a ^ b;
    }
};


/// Bitwise operation left shift
struct bitwiseLeftShift
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
    {
        return a << b;
    }
};

/// Bitwise operation right shift
struct bitwiseRightShift
{
    typedef void RetType;

    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
    {
        return a >> b;
    }
};

/// Mathematical function logarithm
struct logarithm
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        return logf(a) / logf(b);
    }
};

/// Mathematical function arcus tangent
struct arctangent2
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        return atan2f(a, b);
    }
};

/// Mathematical function power
struct power
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        return powf(a, b);
    }
};

/// Mathematical function root
struct root
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        return powf(a, 1.0f / b);
    }
};

/// Mathematical function root
struct roundDecimal
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
    {
        const double multiplier = powf(10.0, b);
        return roundf(a * multiplier) / multiplier;
    }
};

/// Geo-arithmetic function LongitudeToTileX
/// Converts longitude in degrees to tile X at zoom
struct geoLongitudeToTileX
{
    typedef void RetType;

    static constexpr bool isFloatRetType = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U longitude, V zoom, int32_t* errorFlag, T min, T max) const
    {
        return floorf((longitude + 180.0f) / 360.0f * powf(2.0f, zoom));
    }
};

/// Geo-arithmetic function geoLatitudeToTileY
/// Converts latitude in degrees to tile Y at zoom
struct geoLatitudeToTileY
{
    typedef void RetType;

    static constexpr bool isFloatRetType = false;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U latitude, V zoom, int32_t* errorFlag, T min, T max) const
    {
        return (floorf((1.0 - asinhf(tanf(latitude * pi<float> / 180.0f)) / pi<float>) / 2.0 * powf(2.0f, zoom)));
    }
};

/// Geo-arithmetic function geoTileXToLongitude
/// Converts tile X at zoom to longitude in degrees
struct geoTileXToLongitude
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U tileX, V zoom, int32_t* errorFlag, T min, T max) const
    {
        return tileX / powf(2.0f, zoom) * 360.0f - 180.0f;
    }
};

/// Geo-arithmetic function geoTileYToLatitude
/// Converts tile Y at zoom to latitude in degrees
struct geoTileYToLatitude
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U tileY, V zoom, int32_t* errorFlag, T min, T max) const
    {
        float latitudeRad = atanf(sinhf(pi<float> * (1 - 2 * tileY / powf(2.0f, zoom))));
        return latitudeRad * 180.0 / pi<float>;
    }
};

} // namespace ArithmeticOperations
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic operation with column and column
/// (For mod as U and V never use floating point type!)
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template <typename OP, typename T, typename U, typename V>
__global__ void
kernel_arithmetic(T* output, U ACol, V BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] =
            OP{}.template operator()<T, typename std::remove_pointer<U>::type, typename std::remove_pointer<V>::type>(
                maybe_deref(ACol, i), maybe_deref(BCol, i), errorFlag, min, max);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for binary arithmetic functions
template <typename OP, typename T, typename U, typename V, class Enable = void>
class GPUArithmetic
{
public:
    /// Arithmetic operation with two columns
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void Arithmetic(T* output, U ACol, V BCol, int32_t dataElementCount)
    {
        ErrorFlagSwapper errorFlagSwapper;
        kernel_arithmetic<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, ACol, BCol, dataElementCount, errorFlagSwapper.GetFlagPointer(),
                std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        errorFlagSwapper.Swap();
    }
};

template <typename OP, typename T, typename U, typename V>
class GPUArithmetic<OP,
                    T,
                    U,
                    V,
                    typename std::enable_if<std::is_same<typename std::remove_pointer<T>::type, std::string>::value &&
                                            std::is_same<typename std::remove_pointer<U>::type, std::string>::value &&
                                            std::is_same<typename std::remove_pointer<V>::type, std::string>::value>::type>
{
public:
    /// Arithmetic operation with two columns
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void Arithmetic(T* output, U ACol, V BCol, int32_t dataElementCount)
    {
    }
};

template <typename OP, typename T, typename U, typename V>
class GPUArithmetic<OP,
                    T,
                    U,
                    V,
                    typename std::enable_if<std::is_same<typename std::remove_pointer<T>::type, std::string>::value &&
                                            std::is_same<typename std::remove_pointer<U>::type, std::string>::value &&
                                            std::is_integral<typename std::remove_pointer<V>::type>::value>::type>
{
public:
    /// Arithmetic operation with two columns
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void Arithmetic(GPUMemory::GPUString& outCol, GPUMemory::GPUString ACol, V BCol, int32_t dataElementCount)
    {
        if constexpr (std::is_pointer<U>::value && std::is_pointer<V>::value)
        {
            GPUStringBinary::Run<OP, V>(outCol, ACol, dataElementCount, BCol, dataElementCount);
        }
        else if constexpr (std::is_pointer<U>::value && !std::is_pointer<V>::value)
        {
            GPUStringBinary::Run<OP, V>(outCol, ACol, dataElementCount, BCol, 1);
        }
        else if constexpr (!std::is_pointer<U>::value && std::is_pointer<V>::value)
        {
            GPUStringBinary::Run<OP, V>(outCol, ACol, 1, BCol, dataElementCount);
        }
        else
        {
            GPUStringBinary::Run<OP, V>(outCol, ACol, dataElementCount, BCol, 1);
        }
    }
};
