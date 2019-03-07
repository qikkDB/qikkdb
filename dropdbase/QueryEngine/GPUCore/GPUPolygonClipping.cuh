#pragma 

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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


class GPUPolygonIntersect
{

};