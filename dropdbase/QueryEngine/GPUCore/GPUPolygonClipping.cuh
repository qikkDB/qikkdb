#pragma

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"

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

// Struct for the polygon Doubly Linked List construction on the GPU
__host__ __device__ struct PolygonNodeDLL
{
    int32_t poly_group;

    NativeGeoPoint point;

    float linear_distance;
    int32_t is_intersect;

    int32_t next;
    int32_t prev;
    int32_t cross_link;
};

// Data buffers for linked lists of polygons during clipping
__device__ PolygonNodeDLL* poly1List;
__device__ PolygonNodeDLL* poly2List;

template <typename OP>
__global__ void kernel_polygon_clipping(GPUMemory::GPUPolygon out,
                                        GPUMemory::GPUPolygon polygon1,
                                        GPUMemory::GPUPolygon polygon2,
                                        int32_t dataElementCount)
{

}

class GPUPolygonIntersect
{
public:
    template <typename OP>
    static void ColCol(GPUMemory::GPUPolygon out, GPUMemory::GPUPolygon polygon1, GPUMemory::GPUPolygon polygon2, int32_t dataElementCount)
    {
        kernel_polygon_clipping<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                out, polygon1, polygon2, dataElementCount);
        QueryEngineError::setCudaError(cudaGetLastError());
    }
};
