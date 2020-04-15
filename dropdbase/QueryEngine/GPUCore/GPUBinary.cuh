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
#include "GPUConversion.cuh"
#include "GPUStringBinary.cuh"
#include "GPUPolygonClipping.cuh"
#include "GPUFilter.cuh"

namespace ArithmeticOperations
{

/// Arithmetic operation add
struct add
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<L>::value && !std::is_floating_point<R>::value)
        {
            // Check for overflow
            if (((b > R{0}) && (a > (max - b))) || ((b < R{0}) && (a < (min - b))))
            {
                atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                return GetNullConstant<OUT>();
            }
        }
        return a + b;
    }
};

/// Arithmetic operation subtraction
struct sub
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<L>::value && !std::is_floating_point<R>::value)
        {
            // Check for overflow
            if (((b > R{0}) && (a < (min + b))) || ((b < R{0}) && (a > (max + b))))
            {
                atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                return GetNullConstant<OUT>();
            }
        }
        return a - b;
    }
};

/// Arithmetic operation multiply
struct mul
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        // if none of the input operands are float
        if (!std::is_floating_point<L>::value && !std::is_floating_point<R>::value)
        {
            // Check for overflow
            if (a > L{0})
            {
                if (b > R{0})
                {
                    if (a > (max / b))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<OUT>();
                    }
                }
                else
                {
                    if (b < (min / a))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<OUT>();
                    }
                }
            }
            else
            {
                if (b > R{0})
                {
                    if (a < (min / b))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<OUT>();
                    }
                }
                else
                {
                    if ((a != L{0}) && (b < (max / a)))
                    {
                        atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
                        return GetNullConstant<OUT>();
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

    template <typename OUT, typename L, typename R>
    __device__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        if (b == R{0})
        {
            atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
            return GetNullConstant<OUT>();
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

    template <typename OUT, typename L, typename R>
    __device__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        // modulo is not defined for floating point type
        static_assert(!std::is_floating_point<L>::value && !std::is_floating_point<R>::value,
                      "None of the input columns of operation modulo cannot be floating point "
                      "type!");

        // Check for zero division
        if (b == R{0})
        {
            atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
            return GetNullConstant<OUT>();
        }

        return a % b;
    }
};

/// Bitwise operation and
struct bitwiseAnd
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max)
    {
        return a & b;
    }
};

/// Bitwise operation or
struct bitwiseOr
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max)
    {
        return a | b;
    }
};

/// Bitwise operation xor
struct bitwiseXor
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max)
    {
        return a ^ b;
    }
};


/// Bitwise operation left shift
struct bitwiseLeftShift
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max)
    {
        return a << b;
    }
};

/// Bitwise operation right shift
struct bitwiseRightShift
{
    typedef void RetType;

    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max)
    {
        return a >> b;
    }
};

/// Mathematical function logarithm
struct logarithm
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        return logf(a) / logf(b);
    }
};

/// Mathematical function arcus tangent
struct arctangent2
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        return atan2f(a, b);
    }
};

/// Mathematical function power
struct power
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        return powf(a, b);
    }
};

/// Mathematical function root
struct root
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
    {
        return powf(a, 1.0f / b);
    }
};

/// Mathematical function root
struct roundDecimal
{
    typedef void RetType;

    static constexpr bool isFloatRetType = true;
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L a, R b, int32_t* errorFlag, OUT min, OUT max) const
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
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L longitude, R zoom, int32_t* errorFlag, OUT min, OUT max) const
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
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L latitude, R zoom, int32_t* errorFlag, OUT min, OUT max) const
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
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L tileX, R zoom, int32_t* errorFlag, OUT min, OUT max) const
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
    template <typename OUT, typename L, typename R>
    __device__ __host__ OUT operator()(L tileY, R zoom, int32_t* errorFlag, OUT min, OUT max) const
    {
        float latitudeRad = atanf(sinhf(pi<float> * (1 - 2 * tileY / powf(2.0f, zoom))));
        return latitudeRad * 180.0 / pi<float>;
    }
};

} // namespace ArithmeticOperations
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic operation with column and column
/// (For mod as L and R never use floating point type!)
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template <typename OP, typename OUT, typename L, typename R>
__global__ void
kernel_arithmetic(OUT* output, L ACol, R BCol, int32_t dataElementCount, int32_t* errorFlag, OUT min, OUT max)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] =
            OP{}.template operator()<OUT, typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type>(
                maybe_deref(ACol, i), maybe_deref(BCol, i), errorFlag, min, max);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////


template <typename OP>
constexpr bool isFilterOp = std::is_same<OP, FilterConditions::greater>::value ||
                            std::is_same<OP, FilterConditions::greaterEqual>::value ||
                            std::is_same<OP, FilterConditions::less>::value ||
                            std::is_same<OP, FilterConditions::lessEqual>::value ||
                            std::is_same<OP, FilterConditions::equal>::value ||
                            std::is_same<OP, FilterConditions::notEqual>::value ||
                            std::is_same<OP, FilterConditions::logicalAnd>::value ||
                            std::is_same<OP, FilterConditions::logicalOr>::value;

/// Generic binary operations
template <typename OP, typename OUT, typename L, typename R, class Enable = void>
class GPUBinary
{
public:
    /// Arithmetic operation with two columns
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void Binary(OUT* output, L ACol, R BCol, int32_t dataElementCount, int64_t* nullBitMask = nullptr)
    {
        ErrorFlagSwapper errorFlagSwapper;
        kernel_arithmetic<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, ACol, BCol, dataElementCount, errorFlagSwapper.GetFlagPointer(),
                std::numeric_limits<OUT>::min(), std::numeric_limits<OUT>::max());
        errorFlagSwapper.Swap();
    }
};

/// Binary string-string operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                        std::is_same<typename std::remove_pointer<L>::type, std::string>::value &&
                                        std::is_same<typename std::remove_pointer<R>::type, std::string>::value>::type>
{
public:
    static void Binary(GPUMemory::GPUString& outCol,
                       GPUMemory::GPUString ACol,
                       GPUMemory::GPUString BCol,
                       int32_t dataElementCount,
                       int64_t* nullBitMask)
    {
        if constexpr (std::is_pointer<L>::value || std::is_pointer<R>::value)
        {
            GPUStringBinary::Run<OP>(outCol, ACol, std::is_pointer<L>::value, BCol,
                                     std::is_pointer<R>::value, dataElementCount);
        }
        else
        {
            GPUStringBinary::Run<OP>(outCol, ACol, true, BCol, true, dataElementCount);
        }
    }
};

/// Binary string-integer operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                        std::is_same<typename std::remove_pointer<L>::type, std::string>::value &&
                                        std::is_integral<typename std::remove_pointer<R>::type>::value>::type>
{
public:
    static void
    Binary(GPUMemory::GPUString& outCol, GPUMemory::GPUString ACol, R BCol, int32_t dataElementCount, int64_t* nullBitMask)
    {
        if constexpr (std::is_pointer<L>::value && std::is_pointer<R>::value)
        {
            GPUStringBinary::Run<OP, R>(outCol, ACol, dataElementCount, BCol, dataElementCount);
        }
        else if constexpr (std::is_pointer<L>::value && !std::is_pointer<R>::value)
        {
            GPUStringBinary::Run<OP, R>(outCol, ACol, dataElementCount, BCol, 1);
        }
        else if constexpr (!std::is_pointer<L>::value && std::is_pointer<R>::value)
        {
            GPUStringBinary::Run<OP, R>(outCol, ACol, 1, BCol, dataElementCount);
        }
        else
        {
            GPUStringBinary::Run<OP, R>(outCol, ACol, dataElementCount, BCol, 1);
        }
    }
};

/// Numeric-Numeric to Point conversion operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, ColmnarDB::Types::Point>::value &&
                                        std::is_arithmetic<typename std::remove_pointer<L>::type>::value &&
                                        std::is_arithmetic<typename std::remove_pointer<R>::type>::value>::type>
{
public:
    static void Binary(NativeGeoPoint* outCol, L LatCol, R LonCol, int32_t dataElementCount, int64_t* nullBitMask)
    {
        if constexpr (std::is_pointer<L>::value || std::is_pointer<R>::value)
        {
            kernel_convert_lat_lon_to_point<<<Context::getInstance().calcGridDim(dataElementCount),
                                              Context::getInstance().getBlockDim()>>>(outCol, LatCol, LonCol,
                                                                                      dataElementCount);
        }
        else
        {
            GPUMemory::fillArray<NativeGeoPoint>(outCol,
                                                 {static_cast<float>(LatCol), static_cast<float>(LonCol)},
                                                 dataElementCount);
        }
        CheckCudaError(cudaGetLastError());
    }
};

/// Polygon-Polygon operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, ColmnarDB::Types::ComplexPolygon>::value &&
                                        std::is_same<typename std::remove_pointer<L>::type, ColmnarDB::Types::ComplexPolygon>::value &&
                                        std::is_same<typename std::remove_pointer<R>::type, ColmnarDB::Types::ComplexPolygon>::value>::type>
{
public:
    static void Binary(GPUMemory::GPUPolygon& polygonOut,
                       GPUMemory::GPUPolygon& polygonAin,
                       GPUMemory::GPUPolygon& polygonBin,
                       int32_t dataElementCount,
                       int64_t* nullBitMask)
    {
        GPUPolygonClipping::clip<OP>(polygonOut, polygonAin, polygonBin,
                                     std::is_pointer<L>::value ? dataElementCount : 1,
                                     std::is_pointer<R>::value ? dataElementCount : 1);
    }
};

/// Polygon-Point operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, int8_t>::value &&
                                        std::is_same<typename std::remove_pointer<L>::type, ColmnarDB::Types::ComplexPolygon>::value &&
                                        std::is_same<typename std::remove_pointer<R>::type, ColmnarDB::Types::Point>::value>::type>
{
public:
    static void
    Binary(int8_t* outMask,
           GPUMemory::GPUPolygon polygonCol,
           typename std::conditional<std::is_pointer<R>::value, NativeGeoPoint*, NativeGeoPoint>::type geoPointCol,
           int32_t dataElementCount,
           int64_t* nullBitMask)
    {
        if constexpr (std::is_pointer<L>::value || std::is_pointer<R>::value)
        {
            kernel_point_in_polygon<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDim()>>>(
                outMask, polygonCol, std::is_pointer<L>::value ? dataElementCount : 1, geoPointCol,
                std::is_pointer<R>::value ? dataElementCount : 1);
        }
        else
        {
            kernel_point_in_polygon<<<Context::getInstance().calcGridDim(1), Context::getInstance().getBlockDim()>>>(
                outMask, polygonCol, 1, geoPointCol, 1);
            int8_t result;
            GPUMemory::copyDeviceToHost(&result, outMask, 1);
            GPUMemory::memset(outMask, result, dataElementCount);
        }
        CheckCudaError(cudaGetLastError());
    }
};

/// Filter operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<isFilterOp<OP> && std::is_same<typename std::remove_pointer<OUT>::type, int8_t>::value &&
                                        std::is_arithmetic<typename std::remove_pointer<L>::type>::value &&
                                        std::is_arithmetic<typename std::remove_pointer<R>::type>::value>::type>
{
public:
    static void Binary(int8_t* outMask, L ACol, R BCol, int32_t dataElementCount, int64_t* nullBitMask)
    {
        if constexpr (std::is_pointer<L>::value || std::is_pointer<R>::value)
        {
            kernel_filter<OP>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    outMask, ACol, BCol, nullBitMask, dataElementCount);
        }
        else
        {
            GPUMemory::memset(outMask, OP{}(maybe_deref(ACol, 0), maybe_deref(BCol, 0)), dataElementCount);
        }
        CheckCudaError(cudaGetLastError());
    }
};

/// String filter operations
template <typename OP, typename OUT, typename L, typename R>
class GPUBinary<OP,
                OUT,
                L,
                R,
                typename std::enable_if<isFilterOp<OP> && std::is_same<typename std::remove_pointer<OUT>::type, int8_t>::value &&
                                        std::is_same<typename std::remove_pointer<L>::type, std::string>::value &&
                                        std::is_same<typename std::remove_pointer<R>::type, std::string>::value>::type>
{
public:
    static void
    Binary(int8_t* outMask, GPUMemory::GPUString ACol, GPUMemory::GPUString BCol, int32_t dataElementCount, int64_t* nullBitMask)
    {
        kernel_filter_string<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                outMask, ACol, std::is_pointer<L>::value, BCol, std::is_pointer<R>::value,
                nullBitMask, dataElementCount);
        CheckCudaError(cudaGetLastError());

        if constexpr (!std::is_pointer<L>::value && !std::is_pointer<R>::value)
        {
            // Expand mask - copy the one result to whole mask
            int8_t numberFromMask;
            GPUMemory::copyDeviceToHost(&numberFromMask, outMask, 1);
            GPUMemory::memset(outMask, numberFromMask, dataElementCount);
        }
    }
};
