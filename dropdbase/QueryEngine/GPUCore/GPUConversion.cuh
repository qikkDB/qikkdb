#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"

/// Make Point column from latitude and longitude columns
/// <param name="outCol">output NativeGeoPoint column</param>
/// <param name="LatCol">input latitude column</param>
/// <param name="LonCol">input longitude column</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template <typename T, typename U>
__global__ void kernel_convert_lat_lon_to_point(NativeGeoPoint* outCol, T LatCol, U LonCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        outCol[i] = {maybe_deref(LatCol, i), maybe_deref(LonCol, i)};
    }
}

/// Class for converting lat and lon columns to new point column
class GPUConversion
{
public:
    /// Convert two arithmetic (e.g. float) arrays to one NativeGeoPoint array
    /// <param name="outCol">output array of points</param>
    /// <param name="LatCol">input array for latitude</param>
    /// <param name="LonCol">input array for longitude</param>
    /// <param name="dataElementCount">the count of elements in the input block</param>
    template <typename T, typename U>
    static void ConvertColCol(NativeGeoPoint* outCol, T* LatCol, U* LonCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<T>::value, "LatCol must be arithmetic data type");
        static_assert(std::is_arithmetic<U>::value, "LonCol must be arithmetic data type");

        kernel_convert_lat_lon_to_point<<<Context::getInstance().calcGridDim(dataElementCount),
                                          Context::getInstance().getBlockDim()>>>(outCol, LatCol, LonCol,
                                                                                  dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    /// Convert arithmetic (e.g. float) array and arithmetic constant to NativeGeoPoint array
    /// <param name="outCol">output array of points</param>
    /// <param name="LatCol">input array for latitude</param>
    /// <param name="LonCol">input constant for longitude</param>
    /// <param name="dataElementCount">the count of elements in the LatCol array</param>
    template <typename T, typename U>
    static void ConvertColConst(NativeGeoPoint* outCol, T* LatCol, U LonCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<T>::value, "LatCol must be arithmetic data type");
        static_assert(std::is_arithmetic<U>::value, "LonCol must be arithmetic data type");

        kernel_convert_lat_lon_to_point<<<Context::getInstance().calcGridDim(dataElementCount),
                                          Context::getInstance().getBlockDim()>>>(outCol, LatCol, LonCol,
                                                                                  dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    /// Convert arithmetic (e.g. float) constant and array to NativeGeoPoint array
    /// <param name="outCol">output array of points</param>
    /// <param name="LatCol">input constant for latitude</param>
    /// <param name="LonCol">input array for longitude</param>
    /// <param name="dataElementCount">the count of elements in the LonCol array</param>
    template <typename T, typename U>
    static void ConvertConstCol(NativeGeoPoint* outCol, T LatCol, U* LonCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<T>::value, "LatCol must be arithmetic data type");
        static_assert(std::is_arithmetic<U>::value, "LonCol must be arithmetic data type");

        kernel_convert_lat_lon_to_point<<<Context::getInstance().calcGridDim(dataElementCount),
                                          Context::getInstance().getBlockDim()>>>(outCol, LatCol, LonCol,
                                                                                  dataElementCount);
        CheckCudaError(cudaGetLastError());
    }

    /// Convert two arithmetic (e.g. float) constant to NativeGeoPoint array
    /// <param name="outCol">output array of points</param>
    /// <param name="LatCol">input constant for latitude</param>
    /// <param name="LonCol">input constant for longitude</param>
    /// <param name="dataElementCount">the count of elements in the ouptut array</param>
    template <typename T, typename U>
    static void ConvertConstConst(NativeGeoPoint* outCol, T LatCol, U LonCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<T>::value, "LatCol must be arithmetic data type");
        static_assert(std::is_arithmetic<U>::value, "LonCol must be arithmetic data type");
        GPUMemory::fillArray<NativeGeoPoint>(outCol, {LatCol, LonCol}, dataElementCount);
    }
};
