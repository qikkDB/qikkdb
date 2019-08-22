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
        outCol[i] = {static_cast<float>(maybe_deref(LatCol, i)), static_cast<float>(maybe_deref(LonCol, i))};
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
    static void Convert(NativeGeoPoint* outCol, T LatCol, U LonCol, int32_t dataElementCount)
    {
        static_assert(std::is_arithmetic<typename std::remove_pointer<T>::type>::value,
                      "LatCol must be arithmetic data type");
        static_assert(std::is_arithmetic<typename std::remove_pointer<U>::type>::value,
                      "LonCol must be arithmetic data type");
        if (std::is_pointer<T>::value || std::is_pointer<U>::value)
        {
            kernel_convert_lat_lon_to_point<<<Context::getInstance().calcGridDim(dataElementCount),
                                              Context::getInstance().getBlockDim()>>>(outCol, LatCol, LonCol,
                                                                                      dataElementCount);
        }
        else
        {
            // HACK: CUDA doesnt support static_if, so we need to make sure LatCol and LonCol are
            // always values. This branch will never be executed with columns
            GPUMemory::fillArray<NativeGeoPoint>(outCol,
                                                 {static_cast<float>(maybe_deref(LatCol, 0)),
                                                  static_cast<float>(maybe_deref(LonCol, 0))},
                                                 dataElementCount);
        }
        CheckCudaError(cudaGetLastError());
    }
};
