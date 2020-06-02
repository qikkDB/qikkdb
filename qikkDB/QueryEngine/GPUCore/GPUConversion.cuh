#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"
#include "../../Types/Point.pb.h"

namespace ConversionOperations
{
struct latLonToPoint
{
    typedef typename ColmnarDB::Types::Point RetType;
};
} // namespace ConversionOperations

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