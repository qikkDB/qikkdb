#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUArithmetic.cuh"
#include "GPUMemory.cuh"

#include "../../../cub/cub.cuh"
#include "../../NativeGeoPoint.h"
#include "../Context.h"


namespace PolygonFunctions
{
    struct polyIntersect
    {
        __device__ __host__ void operator()() const
        {

        }
    };

    struct polyUnion
    {
        __device__ __host__ void operator()() const
        {

        }
    };
} // namespace PolygonFunctions

class GPUPolygonClipping
{
public:
    template <typename OP>
    static void ColCol(GPUMemory::GPUPolygon& polygonOut,
                       GPUMemory::GPUPolygon polygonAin,
                       GPUMemory::GPUPolygon polygonBin,
                       int32_t dataElementCount)
    {
        
    }
};
