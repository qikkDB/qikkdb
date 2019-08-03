#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "../../../cub/cub.cuh"
#include "GPUTypes.h"

/// Generic agg function functors
namespace AggregationFunctions
{
/// A functor for an aggregate minimum operation
struct min
{
    template <typename T>
    __device__ void operator()(T* a, T b) const
    {
        atomicMin(a, b);
    }

    // Specialized atomicMin for int64_t
    __device__ void operator()(int64_t* a, int64_t b) const
    {
        int64_t old = *a;
        int64_t expected;
        if (old <= b)
        {
            return;
        }

        do
        {
            expected = old;
            uint64_t ret = atomicCAS(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&expected),
                                     *reinterpret_cast<cuUInt64*>(&b));
            old = *(int64_t*)&ret;
        } while (old != expected && old > b);
    }

    // Specialized atomicMin for double
    __device__ void operator()(double* a, double b) const
    {
        double old = *a;
        double expected;
        if (old <= b)
        {
            return;
        }

        do
        {
            expected = old;
            uint64_t ret = atomicCAS(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&expected),
                                     *reinterpret_cast<cuUInt64*>(&b));
            old = *(double*)&ret;
        } while (old != expected && old > b);
    }

    // Specialized atomicMin for floats
    __device__ void operator()(float* a, float b) const
    {
        float old = *a;
        float expected;
        if (old <= b)
        {
            return;
        }

        do
        {
            expected = old;
            int32_t ret = atomicCAS(reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(&expected),
                                    *reinterpret_cast<int32_t*>(&b));
            old = *(float*)&ret;
        } while (old != expected && old > b);
    }

    template <typename OUT, typename IN>
    static void agg(OUT* outValue, IN* ACol, int32_t dataElementCount)
    {
        // Get the buffer size
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        cub::DeviceReduce::Min(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

        // Run minimum reduction - data stays on gpu
        cub::DeviceReduce::Min(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
        GPUMemory::free(tempBuffer);

        cudaDeviceSynchronize();
    }

    template <typename T>
    static constexpr T getInitValue()
    {
        return std::numeric_limits<T>::max();
    }
};

/// A functor for an aggregate maximum operation
struct max
{
    template <typename T>
    __device__ void operator()(T* a, T b) const
    {
        atomicMax(a, b);
    }

    // Specialized atomicMax for int64_t
    __device__ void operator()(int64_t* a, int64_t b) const
    {
        int64_t old = *a;
        int64_t expected;
        if (old >= b)
        {
            return;
        }

        do
        {
            expected = old;
            uint64_t ret = atomicCAS(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&expected),
                                     *reinterpret_cast<cuUInt64*>(&b));
            old = *(int64_t*)&ret;
        } while (old != expected && old < b);
    }

    // Specialized atomicMax for double
    __device__ void operator()(double* a, double b) const
    {
        double old = *a;
        double expected;
        if (old >= b)
        {
            return;
        }

        do
        {
            expected = old;
            uint64_t ret = atomicCAS(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&expected),
                                     *reinterpret_cast<cuUInt64*>(&b));
            old = *(double*)&ret;
        } while (old != expected && old < b);
    }

    // Specialized atomicMax for floats
    __device__ void operator()(float* a, float b) const
    {
        float old = *a;
        float expected;
        if (old >= b)
        {
            return;
        }

        do
        {
            expected = old;
            int32_t ret = atomicCAS(reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(&expected),
                                    *reinterpret_cast<int32_t*>(&b));
            old = *(float*)&ret;
        } while (old != expected && old < b);
    }

    template <typename OUT, typename IN>
    static void agg(OUT* outValue, IN* ACol, int32_t dataElementCount)
    {
        // Get the buffer size
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        cub::DeviceReduce::Max(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

        // Run maximum reduction - data stays on gpu
        cub::DeviceReduce::Max(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
        GPUMemory::free(tempBuffer);

        cudaDeviceSynchronize();
    }

    template <typename T>
    static constexpr T getInitValue()
    {
        return std::numeric_limits<T>::lowest();
    }
};

/// A functor for an aggregate sum operation
struct sum
{
    template <typename T>
    __device__ void operator()(T* a, T b) const
    {
        atomicAdd(a, b);
    }

    __device__ void operator()(int64_t* a, int64_t b) const
    {
        atomicAdd(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&b));
    }

    // atomicAdd double, for CUDA Arch < 600
    __device__ void operator()(double* a, double b) const
    {
        cuUInt64* ptrAsULL = reinterpret_cast<cuUInt64*>(a);
        cuUInt64 old = *ptrAsULL;
        cuUInt64 expected;
        do
        {
            expected = old;
            old = atomicCAS(ptrAsULL, expected, __double_as_longlong(b + __longlong_as_double(expected)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (expected != old);
    }

    template <typename OUT, typename IN>
    static void agg(OUT* outValue, IN* ACol, int32_t dataElementCount)
    {
        // Get the buffer size
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

        // Run sum reduction - data stays on gpu
        cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
        GPUMemory::free(tempBuffer);

        cudaDeviceSynchronize();
    }

    template <typename T>
    static constexpr T getInitValue()
    {
        return T{0};
    }
};

/// A functor for an aggregate averaging operation
struct avg
{
    template <typename T>
    __device__ void operator()(T* a, T b) const
    {
        atomicAdd(a, b);
    }

    __device__ void operator()(int64_t* a, int64_t b) const
    {
        atomicAdd(reinterpret_cast<cuUInt64*>(a), *reinterpret_cast<cuUInt64*>(&b));
    }

    // atomicAdd double, for CUDA Arch < 600
    __device__ void operator()(double* a, double b) const
    {
        cuUInt64* ptrAsULL = reinterpret_cast<cuUInt64*>(a);
        cuUInt64 old = *ptrAsULL;
        cuUInt64 expected;
        do
        {
            expected = old;
            old = atomicCAS(ptrAsULL, expected, __double_as_longlong(b + __longlong_as_double(expected)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (expected != old);
    }

    template <typename OUT, typename IN>
    static void agg(OUT* outValue, IN* ACol, int32_t dataElementCount)
    {
        // Get the buffer size
        void* tempBuffer = nullptr;
        size_t tempBufferSize = 0;
        cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

        // Allocate temporary storage
        GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

        // Run sum reduction - data stays on gpu
        cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
        GPUMemory::free(tempBuffer);

        cudaDeviceSynchronize();

        // Divide the result - calculate the average
        GPUArithmetic::colConst<ArithmeticOperations::div, OUT, OUT, float>(outValue, outValue,
                                                                            static_cast<float>(dataElementCount),
                                                                            1);
    }

    template <typename T>
    static constexpr T getInitValue()
    {
        return T{0};
    }
};

/// A functor for an aggregate counting operation
struct count
{
    template <typename T>
    __device__ void operator()(T* a, T b) const
    {
        // empty
    }

    template <typename OUT, typename IN>
    static void agg(OUT* outValue, IN* ACol, int32_t dataElementCount)
    {
        // TODO, make this function more useful
        OUT temp = dataElementCount;
        GPUMemory::copyHostToDevice(outValue, &temp, 1);
    }

    template <typename T>
    static constexpr T getInitValue()
    {
        return T{0};
    }
};
} // namespace AggregationFunctions