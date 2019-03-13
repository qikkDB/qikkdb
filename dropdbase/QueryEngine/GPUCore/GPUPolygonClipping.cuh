#pragma once

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
__device__ struct PolygonNodeDLL
{
    int32_t poly_group;

    NativeGeoPoint point;

    float linear_distance;
    int32_t is_intersect;

    int32_t next;
    int32_t prev;
    int32_t cross_link;
};

// Data buffers for Polygon Doubly Linked List memory offsets
__device__ int32_t* poly1DLLListStartOffset;
__device__ int32_t* poly2DLLListStartOffset;

// Data buffers for doubly linked lists of polygons during clipping
__device__ PolygonNodeDLL* poly1DLList;
__device__ PolygonNodeDLL* poly2DLList;

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
    static void
    ColCol(GPUMemory::GPUPolygon polygonOut, GPUMemory::GPUPolygon polygon1, GPUMemory::GPUPolygon polygon2, int32_t dataElementCount)
    {
        // Precalcualte the maximal needed size for a doubly linked list as
        // n*k + n + k where n is the number of vertices of polygon 1 and k is the number of
        // vertices of polygon 2 This is a case for one row - for all rows this has to be done
        // dataElementCount times Offsets for the linked list start indexes are needed too - these
        // can be calculated as the prefix sum of doubly linked list sizes The result size of the
        // doubly linked list buffer is a sum of all dll buffers

        // The offset buffers - fil lthem with the prefix sum of input polygon vertices
        GPUMemory::allocAndSet(&poly1DLLListStartOffset, 0, dataElementCount);
        GPUMemory::allocAndSet(&poly2DLLListStartOffset, 0, dataElementCount);


        // The data sandbox for linked lists
        poly1DLList;
        poly2DLList;


        // Alloc space for the doubly linked lists for both polygons in both collumns
        GPUMemory::alloc(&poly1DLList, dataElementCount);
        GPUMemory::alloc(&poly1DLList, dataElementCount);


        kernel_polygon_clipping<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                out, polygon1, polygon2, dataElementCount);
        QueryEngineError::setCudaError(cudaGetLastError());
    }

    static void runDemo()
    {
        // Input polygons
        NativeGeoPoint poly1[] = {{181, 270}, {85, 418},  {171, 477},
                                  {491, 365}, {218, 381}, {458, 260}};
        int32_t complexPolygonIdx1[] = {0};
        int32_t complexPolygonCnt1[] = {1};
        int32_t polygonIdx1[] = {0};
        int32_t polygonCnt1[] = {6};

        NativeGeoPoint poly2[] = {{474, 488}, {659, 363}, {255, 283},
                                  {56, 340},  {284, 488}, {371, 342}};
        int32_t complexPolygonIdx2[] = {0};
        int32_t complexPolygonCnt2[] = {1};
        int32_t polygonIdx2[] = {0};
        int32_t polygonCnt2[] = {6};

        int32_t dataElementCount = 1;

        // Buffers on the GPU
        GPUMemory::GPUPolygon polygon1;
        GPUMemory::GPUPolygon polygon2;
        GPUMemory::GPUPolygon polygonOut;

        // Malloc the buffers
        // Polygon 1
        GPUMemory::alloc(&polygon1.polyPoints, sizeof(poly1) / sizeof(NativeGeoPoint));
        GPUMemory::alloc(&polygon1.polyIdx, sizeof(complexPolygonIdx1) / sizeof(int32_t));
        GPUMemory::alloc(&polygon1.polyCount, sizeof(complexPolygonCnt1) / sizeof(int32_t));
        GPUMemory::alloc(&polygon1.pointIdx, sizeof(polygonIdx1) / sizeof(int32_t));
        GPUMemory::alloc(&polygon1.pointCount, sizeof(polygonCnt1) / sizeof(int32_t));

        // Polygon 2
        GPUMemory::alloc(&polygon2.polyPoints, sizeof(poly2) / sizeof(NativeGeoPoint));
        GPUMemory::alloc(&polygon2.polyIdx, sizeof(complexPolygonIdx2) / sizeof(int32_t));
        GPUMemory::alloc(&polygon2.polyCount, sizeof(complexPolygonCnt2) / sizeof(int32_t));
        GPUMemory::alloc(&polygon2.pointIdx, sizeof(polygonIdx2) / sizeof(int32_t));
        GPUMemory::alloc(&polygon2.pointCount, sizeof(polygonCnt2) / sizeof(int32_t));

        // Copy data to GPU
        // Polygon 1
        GPUMemory::copyHostToDevice(polygon1.polyPoints, poly1, sizeof(poly1) / sizeof(NativeGeoPoint));
        GPUMemory::copyHostToDevice(polygon1.polyIdx, complexPolygonIdx1,
                                    sizeof(complexPolygonIdx1) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon1.polyCount, complexPolygonCnt1,
                                    sizeof(complexPolygonCnt1) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon1.pointIdx, polygonIdx1, sizeof(polygonIdx1) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon1.pointCount, polygonCnt1, sizeof(polygonCnt1) / sizeof(int32_t));

        // Polygon 2
        GPUMemory::copyHostToDevice(polygon2.polyPoints, poly2, sizeof(poly2) / sizeof(NativeGeoPoint));
        GPUMemory::copyHostToDevice(polygon2.polyIdx, complexPolygonIdx2,
                                    sizeof(complexPolygonIdx2) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon2.polyCount, complexPolygonCnt2,
                                    sizeof(complexPolygonCnt2) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon2.pointIdx, polygonIdx2, sizeof(polygonIdx2) / sizeof(int32_t));
        GPUMemory::copyHostToDevice(polygon2.pointCount, polygonCnt2, sizeof(polygonCnt2) / sizeof(int32_t));

        // Launch intersect
        ColCol < PolygonFunctions::polyIntersect>(polygonOut, polygon1, polygon2, dataElementCount);

        // TODO Copy back results

        // TODO Print results

        // Free buffers
        // Polygon 1
        GPUMemory::free(polygon1.polyPoints);
        GPUMemory::free(polygon1.polyIdx);
        GPUMemory::free(polygon1.polyCount);
        GPUMemory::free(polygon1.pointIdx);
        GPUMemory::free(polygon1.pointCount);

        // Polygon 2
        GPUMemory::free(polygon2.polyPoints);
        GPUMemory::free(polygon2.polyIdx);
        GPUMemory::free(polygon2.polyCount);
        GPUMemory::free(polygon2.pointIdx);
        GPUMemory::free(polygon2.pointCount);
    }
};
