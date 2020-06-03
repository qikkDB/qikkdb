#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "ArithmeticUnaryOperations.h"
#include "GPUStringUnary.cuh"
#include "DateOperations.h"
#include "GPUFilterConditions.cuh"
#include "GPUDate.cuh"
#include "GPUCast.cuh"
#include "GPULogic.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic unary operation with column and column
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template <typename OP, typename OUT, typename IN>
__global__ void kernel_arithmetic_unary(OUT* output, IN ACol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] =
            OP{}.template operator()<OUT, typename std::remove_pointer<IN>::type>(maybe_deref(ACol, i));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic unary operations
template <typename OP, typename OUT, typename IN, class Enable = void>
class GPUUnary
{
public:
    /// Arithmetic unary operation with values from column
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
    /// <param name="output">output GPU buffer</param>
    /// <param name="ACol">buffer with operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
    static void Unary(OUT* output, IN ACol, int32_t dataElementCount, nullmask_t* nullBitMask = nullptr)
    {
        kernel_arithmetic_unary<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, ACol, dataElementCount);
    }
};

/// String input - string output operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                       std::is_same<typename std::remove_pointer<IN>::type, std::string>::value>::type>
{
public:
    /// String unary operations which return string, for column
    /// <param name="output">output string column</param>
    /// <param name="inCol">input string column (GPUString)</param>
    /// <param name="dataElementCount">input string count</param>
    static void
    Unary(GPUMemory::GPUString& output, GPUMemory::GPUString inCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        output = OP{}(inCol, dataElementCount);
    }
};

/// String input - numeric output operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, StringUnaryNumericOperations::len>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, int32_t>::value &&
                                       std::is_same<typename std::remove_pointer<IN>::type, std::string>::value>::type>
{
public:
    /// String unary operations which return string, for column
    /// <param name="output">output string column</param>
    /// <param name="inCol">input string column (GPUString)</param>
    /// <param name="dataElementCount">input string count</param>
    static void Unary(int32_t* outCol, GPUMemory::GPUString inCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        if constexpr (std::is_pointer<IN>::value)
        {
            Context& context = Context::getInstance();
            kernel_lengths_from_indices<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                outCol, inCol.stringIndices, dataElementCount);
            CheckCudaError(cudaGetLastError());
        }
        else
        {
            // Copy single index to host
            int64_t hostIndex;
            GPUMemory::copyDeviceToHost(&hostIndex, inCol.stringIndices, 1);
            // Cast to int32 and copy to return result
            int32_t length = static_cast<int32_t>(hostIndex);
            GPUMemory::copyHostToDevice(outCol, &length, 1);
        }
    }
};

/// Date to string cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, DateOperations::toString>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                       std::is_same<typename std::remove_pointer<IN>::type, int64_t>::value>::type>
{
public:
    static void Unary(GPUMemory::GPUString& output, IN dateTimeCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        static_assert(std::is_same<typename std::remove_pointer<IN>::type, int64_t>::value,
                      "DateTime can only be extracted from int64 columns");

        if (dataElementCount > 0)
        {

            // Length of date format "YYYY-MM-DD HH:MM:SS" = 19

            const int32_t dateFormatLength = 19;

            GPUMemory::alloc(&(output.stringIndices), dataElementCount);

            cuda_ptr<int32_t> lengths(dataElementCount);
            kernel_fill_array<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(lengths.get(), dateFormatLength,
                                                                        dataElementCount);

            GPUReconstruct::PrefixSum(output.stringIndices, lengths.get(), dataElementCount);

            int64_t totalCharCount;
            GPUMemory::copyDeviceToHost(&totalCharCount, output.stringIndices + dataElementCount - 1, 1);
            GPUMemory::alloc(&(output.allChars), totalCharCount);

            cuda_ptr<int32_t> years(dataElementCount);
            cuda_ptr<int32_t> months(dataElementCount);
            cuda_ptr<int32_t> days(dataElementCount);
            cuda_ptr<int32_t> hours(dataElementCount);
            cuda_ptr<int32_t> minutes(dataElementCount);
            cuda_ptr<int32_t> seconds(dataElementCount);


            kernel_arithmetic_unary<DateOperations::year, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    years.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::month, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    months.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::day, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    days.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::hour, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    hours.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::minute, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    minutes.get(), dateTimeCol, dataElementCount);

            kernel_arithmetic_unary<DateOperations::second, int32_t, IN>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    seconds.get(), dateTimeCol, dataElementCount);

            kernel_fill_date_string<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDim()>>>(output, years.get(),
                                                                              months.get(), days.get(),
                                                                              hours.get(), minutes.get(),
                                                                              seconds.get(), dataElementCount);
            cudaDeviceSynchronize();
            CheckCudaError(cudaGetLastError());
        }
        else
        {
            output.allChars = nullptr;
            output.allChars = nullptr;
        }
    }
};

/// Numeric to string cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, CastOperations::toString>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                       std::is_arithmetic<typename std::remove_pointer<IN>::type>::value>::type>
{
public:
    static void Unary(GPUMemory::GPUString& outCol, IN inCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        GPUCast::CastNumericToString(outCol, inCol, dataElementCount);
    }
};

/// Point to string cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, CastOperations::toString>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                       std::is_same<typename std::remove_pointer<IN>::type, ColmnarDB::Types::Point>::value>::type>
{
public:
    static void
    Unary(GPUMemory::GPUString& outCol,
          typename std::conditional<std::is_pointer<IN>::value, NativeGeoPoint*, NativeGeoPoint>::type inCol,
          int32_t dataElementCount,
          nullmask_t* nullBitMask)
    {
        GPUReconstruct::ConvertPointColToWKTCol(outCol, inCol, dataElementCount);
    }
};

/// Polygon to string cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, CastOperations::toString>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, std::string>::value &&
                                       std::is_same<typename std::remove_pointer<IN>::type, ColmnarDB::Types::ComplexPolygon>::value>::type>
{
public:
    static void
    Unary(GPUMemory::GPUString& outCol, GPUMemory::GPUPolygon inCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        GPUReconstruct::ConvertPolyColToWKTCol(outCol, inCol, dataElementCount);
    }
};

/// Numeric to numeric cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, CastOperations::toNumeric<OUT>>::value &&
                                       std::is_arithmetic<typename std::remove_pointer<OUT>::type>::value &&
                                       std::is_arithmetic<typename std::remove_pointer<IN>::type>::value>::type>
{
public:
    static void Unary(OUT* outCol, IN inCol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        kernel_cast_numeric<<<Context::getInstance().calcGridDim(dataElementCount),
                              Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }
};

/// String to numeric/point cast operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, CastOperations::toNumeric<OUT>>::value &&
                                       (std::is_arithmetic<typename std::remove_pointer<OUT>::type>::value ||
                                        std::is_same<typename std::remove_pointer<OUT>::type, ColmnarDB::Types::Point>::value) &&
                                       std::is_same<typename std::remove_pointer<IN>::type, std::string>::value>::type>
{
public:
    static void
    Unary(typename std::conditional<std::is_same<typename std::remove_pointer<OUT>::type, ColmnarDB::Types::Point>::value, NativeGeoPoint*, OUT*>::type outCol,
          GPUMemory::GPUString inCol,
          int32_t dataElementCount,
          nullmask_t* nullBitMask)
    {
        kernel_cast_string<<<Context::getInstance().calcGridDim(dataElementCount),
                             Context::getInstance().getBlockDim()>>>(outCol, inCol, dataElementCount);

        CheckCudaError(cudaGetLastError());
    }
};

/// Logical not operations
template <typename OP, typename OUT, typename IN>
class GPUUnary<OP,
               OUT,
               IN,
               typename std::enable_if<std::is_same<OP, FilterConditions::logicalNot>::value &&
                                       std::is_same<typename std::remove_pointer<OUT>::type, int8_t>::value &&
                                       std::is_arithmetic<typename std::remove_pointer<IN>::type>::value>::type>
{
public:
    static void Unary(int8_t* outCol, IN ACol, int32_t dataElementCount, nullmask_t* nullBitMask)
    {
        kernel_operator_not<<<Context::getInstance().calcGridDim(dataElementCount),
                              Context::getInstance().getBlockDim()>>>(outCol, ACol, nullBitMask, dataElementCount);

        CheckCudaError(cudaGetLastError());
    }
};
