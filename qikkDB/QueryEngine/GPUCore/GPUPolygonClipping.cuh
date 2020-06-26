#pragma once

#include <cstdio>
#include <iostream>
#include <cstdint>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"
#include "GPUReconstruct.cuh"

#include "../../../cub/cub.cuh"
#include "../../NativeGeoPoint.h"
#include "../../Types/ComplexPolygon.pb.h"
#include "../Context.h"
#include "cuda_ptr.h"


namespace PolygonFunctions
{
struct polyIntersect
{
    typedef QikkDB::Types::ComplexPolygon RetType;

    __device__ __host__ void operator()(bool turnTable[2]) const
    {
        turnTable[0] = true;
        turnTable[1] = true;
    }
};

struct polyUnion
{
    typedef QikkDB::Types::ComplexPolygon RetType;

    __device__ __host__ void operator()(bool turnTable[2]) const
    {
        turnTable[0] = false;
        turnTable[1] = false;
    }
};

struct contains
{
    typedef int8_t RetType;
};
} // namespace PolygonFunctions

// A point in the linked list of polygons
struct LLPolyVertex
{
    NativeGeoPoint vertex; // The vertex coordinates

    // One variable to represent the linked list flags
    // hasIntersections - Tells if a sub-polygon from the complex polygon to which this vertex
    // belongs has an intersection with the other complex polygon isIntersection - Is this an
    // intersection or a polygon vertex isValidIntersection - Is this a valid interection ? ( does
    // the point lie between the crossing lines) isEntry - Is this an entry (true) or an exit
    // (false) to the other polygon wasProcessed - Was the vertex processed already during the
    // result clipping | empty | empty | empty | hasIntersections | isIntersection |
    // isValidIntersection | isEntry | wasProcessed |
    uint8_t llflags;

    float distanceAlongA; // Distance of the intersection from the beginning of the first line
    float distanceAlongB; // Distance of the intersection from the beginning of the second line

    int32_t prevIdx; // Index of the previous member in the ll
    int32_t nextIdx; // Index of the next member in the ll
    int32_t crossIdx; // Index in the other complex polygon for cross linking during traversal

    // Getting and setting the bit flags for the polygon linked list methods
    // Getters
    __device__ bool GetHasIntersections();
    __device__ bool GetIsIntersection();
    __device__ bool GetIsValidIntersection();
    __device__ bool GetIsEntry();
    __device__ bool GetWasProcessed();

    // Setters
    __device__ void SetHasIntersections(bool flag);
    __device__ void SetIsIntersection(bool flag);
    __device__ void SetIsValidIntersection(bool flag);
    __device__ void SetIsEntry(bool flag);
    __device__ void SetWasProcessed(bool flag);
};

// Calcualte an intersection point between two lines
__device__ LLPolyVertex calc_intersect(NativeGeoPoint sA, NativeGeoPoint eA, NativeGeoPoint sB, NativeGeoPoint eB);

// Calculate the required sizes of the linked lists
// This is clacluated as n + k + intersection_count where n and k
// are the counts of vertices of the complex polygons A and B
__global__ void kernel_calc_ll_buffers_size(int32_t* llPolygonABufferSizes,
                                            int32_t* llPolygonBBufferSizes,
                                            int8_t* PolygonAIntersectionPresenceFlags,
                                            int8_t* PolygonBIntersectionPresenceFlags,
                                            GPUMemory::GPUPolygon polygonA,
                                            GPUMemory::GPUPolygon polygonB,
                                            bool isAConst,
                                            bool isBConst,
                                            int32_t dataElementCount);

// Build the linked lists from the polygons
__global__ void kernel_build_ll(LLPolyVertex* llPolygonBuffers,
                                GPUMemory::GPUPolygon polygon,
                                int32_t* llPolygonBufferSizesPrefixSum,
                                int8_t* PolygonIntersectionPresenceFlags,
                                bool isConst,
                                int32_t dataElementCount);

// Insert the intersections into the linked lists and cross link the intersections between complex polygons
__global__ void kernel_add_and_crosslink_intersections_to_ll(LLPolyVertex* llPolygonABuffers,
                                                             LLPolyVertex* llPolygonBBuffers,
                                                             GPUMemory::GPUPolygon polygonA,
                                                             GPUMemory::GPUPolygon polygonB,
                                                             int32_t* llPolygonABufferSizesPrefixSum,
                                                             int32_t* llPolygonBBufferSizesPrefixSum,
                                                             bool isAConst,
                                                             bool isBConst,
                                                             int32_t dataElementCount);

// Check if a point is withing a complex polygon at a given index
__device__ bool
is_point_in_complex_polygon_at(NativeGeoPoint geoPoint, GPUMemory::GPUPolygon polygon, int32_t idx);

// Decide which intersection points are entry points and whoch ones are exit points and label them accordingly
__global__ void kernel_label_intersections(LLPolyVertex* llPolygonBuffers,
                                           GPUMemory::GPUPolygon polygonPrimary,
                                           GPUMemory::GPUPolygon polygonSecondary,
                                           int32_t* llPolygonBufferSizesPrefixSum,
                                           bool isPrimaryConst,
                                           bool isSecondaryConst,
                                           int32_t dataElementCount);

template <typename T>
__global__ void
kernel_point_in_polygon(int8_t* outMask, GPUMemory::GPUPolygon polygonCol, int32_t polygonCount, T geoPointCol, int32_t pointCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < (pointCount > polygonCount ? pointCount : polygonCount); i += stride)
    {
        int32_t polyIdx = (polygonCount == 1) ? 0 : i;
        outMask[i] = is_point_in_complex_polygon_at(maybe_deref(geoPointCol, i), polygonCol, polyIdx);
    }
}

// Clip the polygons in different phases
template <typename OP>
__device__ void clip_polygons(int32_t* polyCount,
                              int32_t* polyIdx,
                              int32_t* pointCount,
                              int32_t* pointIdx,
                              NativeGeoPoint* polyPoints,
                              LLPolyVertex* llPolygonABuffers,
                              LLPolyVertex* llPolygonBBuffers,
                              GPUMemory::GPUPolygon polygonA,
                              GPUMemory::GPUPolygon polygonB,
                              int32_t* llPolygonABufferSizesPrefixSum,
                              int32_t* llPolygonBBufferSizesPrefixSum,
                              int8_t* PolygonAIntersectionPresenceFlags,
                              int8_t* PolygonBIntersectionPresenceFlags,
                              bool isAConst,
                              bool isBConst,
                              int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t iAIdx = isAConst ? 0 : i;
        int32_t iBIdx = isBConst ? 0 : i;

        // Get the complex polygon vertex counts n and k
        const int32_t n = polygonA.TotalPointCountAt(iAIdx);
        const int32_t k = polygonB.TotalPointCountAt(iBIdx);

        const int32_t llABegOffset = ((i == 0) ? 0 : llPolygonABufferSizesPrefixSum[i - 1]);

        const int32_t begIdxA = llABegOffset + n;
        const int32_t endIdxA = llPolygonABufferSizesPrefixSum[i];

        const int32_t begIdxB = ((i == 0) ? 0 : llPolygonBBufferSizesPrefixSum[i - 1]) + k;
        const int32_t endIdxB = llPolygonBBufferSizesPrefixSum[i];

        // Fill the turn table according to clipping operation
        bool turnTable[2];
        OP{}(turnTable);

        //////////////////////////////////////////////////////////////////////////////
        // Reconstruct the NON-intersecting poylgon result

        // Calculate the non intersecting component counts
        int32_t nonIntersectComponentCount = 0;

        // Add the non intersecting polygons to the result
        const int32_t polyIdxA = polygonA.PolyIdxAt(iAIdx);
        const int32_t polyCountA = polygonA.PolyCountAt(iAIdx);
        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            const int32_t pointIdxA = polygonA.PointIdxAt(a);
            const int32_t pointCountA = polygonA.PointCountAt(a);
            if (PolygonAIntersectionPresenceFlags[isAConst ? a + i * dataElementCount : a] == 0)
            {
                bool isAinB = is_point_in_complex_polygon_at(polygonA.polyPoints[pointIdxA], polygonB, iBIdx);
                if (isAinB == turnTable[0])
                {
                    int32_t noIntersectSubComponentCount = 0;

                    for (int32_t pointA = pointIdxA; pointA < (pointIdxA + pointCountA); pointA++)
                    {
                        if (polyCount && pointCount && polyPoints)
                        {
                            const int32_t polyIdxTemp = i;
                            const int32_t pointIdxTemp =
                                (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) + nonIntersectComponentCount);
                            const int32_t polyPointIdxTemp =
                                (((pointIdxTemp == 0) ? 0 : pointIdx[pointIdxTemp - 1]) + noIntersectSubComponentCount);

                            polyPoints[polyPointIdxTemp] = polygonA.polyPoints[pointA];
                        }
                        noIntersectSubComponentCount++;
                    }

                    if (polyCount && pointCount)
                    {
                        const int32_t polyIdxTemp = i;
                        const int32_t pointIdxTemp =
                            (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) + nonIntersectComponentCount);
                        pointCount[pointIdxTemp] = noIntersectSubComponentCount;
                    }

                    nonIntersectComponentCount++;
                }
            }
        }

        const int32_t polyIdxB = polygonB.PolyIdxAt(iBIdx);
        const int32_t polyCountB = polygonB.PolyCountAt(iBIdx);
        for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
        {
            const int32_t pointIdxB = polygonB.PointIdxAt(b);
            const int32_t pointCountB = polygonB.PointCountAt(b);

            if (PolygonBIntersectionPresenceFlags[isBConst ? b + i * dataElementCount : b] == 0)
            {
                bool isBinA = is_point_in_complex_polygon_at(polygonB.polyPoints[pointIdxB], polygonA, iAIdx);
                if (isBinA == turnTable[1])
                {
                    int32_t noIntersectSubComponentCount = 0;

                    for (int32_t pointB = pointIdxB; pointB < (pointIdxB + pointCountB); pointB++)
                    {
                        if (polyCount && pointCount && polyPoints)
                        {
                            const int32_t polyIdxTemp = i;
                            const int32_t pointIdxTemp =
                                (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) + nonIntersectComponentCount);
                            const int32_t polyPointIdxTemp =
                                (((pointIdxTemp == 0) ? 0 : pointIdx[pointIdxTemp - 1]) + noIntersectSubComponentCount);

                            polyPoints[polyPointIdxTemp] = polygonB.polyPoints[pointB];
                        }
                        noIntersectSubComponentCount++;
                    }

                    if (polyCount && pointCount)
                    {
                        const int32_t polyIdxTemp = i;
                        const int32_t pointIdxTemp =
                            (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) + nonIntersectComponentCount);
                        pointCount[pointIdxTemp] = noIntersectSubComponentCount;
                    }

                    nonIntersectComponentCount++;
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////
        // Reconstruct the intersecting poylgon result

        // Traverse the linked list and calcualte the intersecting component counts
        int32_t turnNumber = 0;
        int32_t intersectComponentCount = 0;
        LLPolyVertex* llPolygonBuffersTable[2] = {llPolygonABuffers, llPolygonBBuffers};
        for (int32_t point = begIdxA; point < endIdxA; point++)
        {
            if (llPolygonBuffersTable[turnNumber][point].GetWasProcessed() == false)
            {
                // Calculate the sub component count
                int32_t IntersectSubComponentCount = 0;

                int32_t nextIdx = point;
                do
                {
                    llPolygonBuffersTable[turnNumber][nextIdx].SetWasProcessed(true);
                    llPolygonBuffersTable[1 - turnNumber][llPolygonBuffersTable[turnNumber][nextIdx].crossIdx]
                        .SetWasProcessed(true);

                    bool forward =
                        (llPolygonBuffersTable[turnNumber][nextIdx].GetIsEntry() == turnTable[turnNumber]);
                    do
                    {
                        // Write the output point
                        if (polyCount && pointCount && polyPoints)
                        {
                            const int32_t polyIdxTemp = i;
                            const int32_t pointIdxTemp =
                                (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) +
                                 nonIntersectComponentCount + intersectComponentCount);
                            const int32_t polyPointIdxTemp =
                                (((pointIdxTemp == 0) ? 0 : pointIdx[pointIdxTemp - 1]) + IntersectSubComponentCount);
                            polyPoints[polyPointIdxTemp] =
                                llPolygonBuffersTable[turnNumber][nextIdx].vertex;
                        }

                        if (forward)
                        {
                            nextIdx = llPolygonBuffersTable[turnNumber][nextIdx].nextIdx;
                        }
                        else
                        {
                            nextIdx = llPolygonBuffersTable[turnNumber][nextIdx].prevIdx;
                        }

                        IntersectSubComponentCount++;
                    } while (!llPolygonBuffersTable[turnNumber][nextIdx].GetIsIntersection());

                    nextIdx = llPolygonBuffersTable[turnNumber][nextIdx].crossIdx;
                    turnNumber = 1 - turnNumber;
                } while (!llPolygonBuffersTable[turnNumber][nextIdx].GetWasProcessed());

                if (polyCount && pointCount)
                {
                    const int32_t polyIdxTemp = i;
                    const int32_t pointIdxTemp = (((polyIdxTemp == 0) ? 0 : polyIdx[polyIdxTemp - 1]) +
                                                  nonIntersectComponentCount + intersectComponentCount);
                    pointCount[pointIdxTemp] = IntersectSubComponentCount;
                }

                intersectComponentCount++;
            }
        }

        // Write the intersecting polygons result sizes
        if (polyCount)
        {
            const int32_t polyIdxTemp = i;
            polyCount[polyIdxTemp] = nonIntersectComponentCount + intersectComponentCount;
        }

        //////////////////////////////////////////////////////////////////////////////
        // Reset the processed flags for the next reconstruction operation
        for (int32_t pointA = begIdxA; pointA < endIdxA; pointA++)
        {
            llPolygonABuffers[pointA].SetWasProcessed(false);
        }

        for (int32_t pointB = begIdxB; pointB < endIdxB; pointB++)
        {
            llPolygonBBuffers[pointB].SetWasProcessed(false);
        }
    }
}

// A set of methods for clipping
// Reconstruct the polyIdx of the result polygon
template <typename OP>
__global__ void kernel_clip_polyIdx(int32_t* polyCount,
                                    LLPolyVertex* llPolygonABuffers,
                                    LLPolyVertex* llPolygonBBuffers,
                                    GPUMemory::GPUPolygon polygonA,
                                    GPUMemory::GPUPolygon polygonB,
                                    int32_t* llPolygonABufferSizesPrefixSum,
                                    int32_t* llPolygonBBufferSizesPrefixSum,
                                    int8_t* PolygonAIntersectionPresenceFlags,
                                    int8_t* PolygonBIntersectionPresenceFlags,
                                    bool isAConst,
                                    bool isBConst,
                                    int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, nullptr, nullptr, nullptr, nullptr, llPolygonABuffers,
                      llPolygonBBuffers, polygonA, polygonB, llPolygonABufferSizesPrefixSum,
                      llPolygonBBufferSizesPrefixSum, PolygonAIntersectionPresenceFlags,
                      PolygonBIntersectionPresenceFlags, isAConst, isBConst, dataElementCount);
}

// Reconstruct the pointIdx of the result polygon
template <typename OP>
__global__ void kernel_clip_pointIdx(int32_t* polyCount,
                                     int32_t* polyIdx,
                                     int32_t* pointCount,
                                     LLPolyVertex* llPolygonABuffers,
                                     LLPolyVertex* llPolygonBBuffers,
                                     GPUMemory::GPUPolygon polygonA,
                                     GPUMemory::GPUPolygon polygonB,
                                     int32_t* llPolygonABufferSizesPrefixSum,
                                     int32_t* llPolygonBBufferSizesPrefixSum,
                                     int8_t* PolygonAIntersectionPresenceFlags,
                                     int8_t* PolygonBIntersectionPresenceFlags,
                                     bool isAConst,
                                     bool isBConst,
                                     int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, polyIdx, pointCount, nullptr, nullptr, llPolygonABuffers,
                      llPolygonBBuffers, polygonA, polygonB, llPolygonABufferSizesPrefixSum,
                      llPolygonBBufferSizesPrefixSum, PolygonAIntersectionPresenceFlags,
                      PolygonBIntersectionPresenceFlags, isAConst, isBConst, dataElementCount);
}

// Reconstruc the polyPoints of the result polygon
template <typename OP>
__global__ void kernel_clip_polyPoints(int32_t* polyCount,
                                       int32_t* polyIdx,
                                       int32_t* pointCount,
                                       int32_t* pointIdx,
                                       NativeGeoPoint* polyPoints,
                                       LLPolyVertex* llPolygonABuffers,
                                       LLPolyVertex* llPolygonBBuffers,
                                       GPUMemory::GPUPolygon polygonA,
                                       GPUMemory::GPUPolygon polygonB,
                                       int32_t* llPolygonABufferSizesPrefixSum,
                                       int32_t* llPolygonBBufferSizesPrefixSum,
                                       int8_t* PolygonAIntersectionPresenceFlags,
                                       int8_t* PolygonBIntersectionPresenceFlags,
                                       bool isAConst,
                                       bool isBConst,
                                       int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, polyIdx, pointCount, pointIdx, polyPoints, llPolygonABuffers,
                      llPolygonBBuffers, polygonA, polygonB, llPolygonABufferSizesPrefixSum,
                      llPolygonBBufferSizesPrefixSum, PolygonAIntersectionPresenceFlags,
                      PolygonBIntersectionPresenceFlags, isAConst, isBConst, dataElementCount);
}

class GPUPolygonClipping
{
public:
    template <typename OP>
    static void clip(GPUMemory::GPUPolygon& polygonOut,
                     GPUMemory::GPUPolygon& polygonAin,
                     GPUMemory::GPUPolygon& polygonBin,
                     int32_t dataElementCountA,
                     int32_t dataElementCountB)
    {
        // Choose the kernel grid size
        const int32_t dataElementCount = std::max(dataElementCountA, dataElementCountB);

        const bool isAConst = (dataElementCountA == 1);
        const bool isBConst = (dataElementCountB == 1);

        // Create buffers for the linked lists
        // Allocate size buffers
        cuda_ptr<int32_t> llPolygonABufferSizes(dataElementCount);
        cuda_ptr<int32_t> llPolygonBBufferSizes(dataElementCount);

        // Allocate polygon intersection flags buffers
        // These buffers indicate if a polygon from a complex polygon A has a intersection with complex polygon B
        int32_t PolygonAIntersectionPresenceFlagsCount;
        int32_t PolygonBIntersectionPresenceFlagsCount;

        GPUMemory::copyDeviceToHost(&PolygonAIntersectionPresenceFlagsCount,
                                    polygonAin.polyIdx + dataElementCountA - 1, 1);
        GPUMemory::copyDeviceToHost(&PolygonBIntersectionPresenceFlagsCount,
                                    polygonBin.polyIdx + dataElementCountB - 1, 1);

        // Multiply the size of the intersect count tables if a input col is const by the row count - data element count
        if (isAConst)
        {
            PolygonAIntersectionPresenceFlagsCount *= dataElementCountB;
        }

        if (isBConst)
        {
            PolygonBIntersectionPresenceFlagsCount *= dataElementCountA;
        }

        cuda_ptr<int8_t> PolygonAIntersectionPresenceFlags(PolygonAIntersectionPresenceFlagsCount, 0);
        cuda_ptr<int8_t> PolygonBIntersectionPresenceFlags(PolygonBIntersectionPresenceFlagsCount, 0);

        // Calcualte the required buffer sizes and the presence of the intersect flags
        kernel_calc_ll_buffers_size<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDimPoly()>>>(
            llPolygonABufferSizes.get(), llPolygonBBufferSizes.get(),
            PolygonAIntersectionPresenceFlags.get(), PolygonBIntersectionPresenceFlags.get(),
            polygonAin, polygonBin, isAConst, isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Calculate the inclusive prefix sum for the ll buffer sizes counters for adressing purpose
        cuda_ptr<int32_t> llPolygonABufferSizesPrefixSum(dataElementCount);
        cuda_ptr<int32_t> llPolygonBBufferSizesPrefixSum(dataElementCount);

        GPUReconstruct::PrefixSum(llPolygonABufferSizesPrefixSum.get(), llPolygonABufferSizes.get(), dataElementCount);
        GPUReconstruct::PrefixSum(llPolygonBBufferSizesPrefixSum.get(), llPolygonBBufferSizes.get(), dataElementCount);

        // Copy back the total size of the ll buffers
        int32_t llPolygonABufferSizesTotal;
        int32_t llPolygonBBufferSizesTotal;

        GPUMemory::copyDeviceToHost(&llPolygonABufferSizesTotal,
                                    llPolygonABufferSizesPrefixSum.get() + dataElementCount - 1, 1);
        GPUMemory::copyDeviceToHost(&llPolygonBBufferSizesTotal,
                                    llPolygonBBufferSizesPrefixSum.get() + dataElementCount - 1, 1);

        // Alloc the linked list buffers for the polygon clipping
        cuda_ptr<LLPolyVertex> llPolygonABuffers(llPolygonABufferSizesTotal);
        cuda_ptr<LLPolyVertex> llPolygonBBuffers(llPolygonBBufferSizesTotal);

        // Transform the complex polygons into linked lists
        // A polygon
        kernel_build_ll<<<Context::getInstance().calcGridDim(dataElementCount),
                          Context::getInstance().getBlockDim()>>>(llPolygonABuffers.get(), polygonAin,
                                                                  llPolygonABufferSizesPrefixSum.get(),
                                                                  PolygonAIntersectionPresenceFlags.get(),
                                                                  isAConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // B polygon
        kernel_build_ll<<<Context::getInstance().calcGridDim(dataElementCount),
                          Context::getInstance().getBlockDim()>>>(llPolygonBBuffers.get(), polygonBin,
                                                                  llPolygonBBufferSizesPrefixSum.get(),
                                                                  PolygonBIntersectionPresenceFlags.get(),
                                                                  isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Insert the intersections into the linked lists and cross link the intersections for intersect/union traversal
        kernel_add_and_crosslink_intersections_to_ll<<<Context::getInstance().calcGridDim(dataElementCount),
                                                       Context::getInstance().getBlockDim()>>>(
            llPolygonABuffers.get(), llPolygonBBuffers.get(), polygonAin, polygonBin,
            llPolygonABufferSizesPrefixSum.get(), llPolygonBBufferSizesPrefixSum.get(), isAConst,
            isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Decide which intersection points are entry points and which ones are exit points
        // and label them accordingly
        kernel_label_intersections<<<Context::getInstance().calcGridDim(dataElementCount),
                                     Context::getInstance().getBlockDim()>>>(
            llPolygonABuffers.get(), polygonAin, polygonBin, llPolygonABufferSizesPrefixSum.get(),
            isAConst, isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        kernel_label_intersections<<<Context::getInstance().calcGridDim(dataElementCount),
                                     Context::getInstance().getBlockDim()>>>(
            llPolygonBBuffers.get(), polygonBin, polygonAin, llPolygonBBufferSizesPrefixSum.get(),
            isBConst, isAConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Process the polyIdx array
        cuda_ptr<int32_t> polyCount(dataElementCount);

        kernel_clip_polyIdx<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), llPolygonABuffers.get(), llPolygonBBuffers.get(), polygonAin,
                polygonBin, llPolygonABufferSizesPrefixSum.get(),
                llPolygonBBufferSizesPrefixSum.get(), PolygonAIntersectionPresenceFlags.get(),
                PolygonBIntersectionPresenceFlags.get(), isAConst, isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUMemory::alloc(&(polygonOut.polyIdx), dataElementCount);
        GPUReconstruct::PrefixSum(polygonOut.polyIdx, polyCount.get(), dataElementCount);

        // Retrieve the pointIdx array length
        int32_t pointIdxSize;
        GPUMemory::copyDeviceToHost(&pointIdxSize, polygonOut.polyIdx + dataElementCount - 1, 1);

        // Handle a completely empty result set
        if (pointIdxSize == 0)
        {
            polygonOut.pointIdx = nullptr;
            polygonOut.polyPoints = nullptr;

            return;
        }

        // Process the pointIdx array
        cuda_ptr<int32_t> pointCount(pointIdxSize);

        kernel_clip_pointIdx<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), polygonOut.polyIdx, pointCount.get(), llPolygonABuffers.get(),
                llPolygonBBuffers.get(), polygonAin, polygonBin, llPolygonABufferSizesPrefixSum.get(),
                llPolygonBBufferSizesPrefixSum.get(), PolygonAIntersectionPresenceFlags.get(),
                PolygonBIntersectionPresenceFlags.get(), isAConst, isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUMemory::alloc(&(polygonOut.pointIdx), pointIdxSize);
        GPUReconstruct::PrefixSum(polygonOut.pointIdx, pointCount.get(), pointIdxSize);

        // Retrieve the polyPoints array length
        int32_t polyPointsSize;
        GPUMemory::copyDeviceToHost(&polyPointsSize, polygonOut.pointIdx + pointIdxSize - 1, 1);

        // Process the polyPoints array
        GPUMemory::alloc(&(polygonOut.polyPoints), polyPointsSize);

        kernel_clip_polyPoints<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), polygonOut.polyIdx, pointCount.get(), polygonOut.pointIdx,
                polygonOut.polyPoints, llPolygonABuffers.get(), llPolygonBBuffers.get(), polygonAin,
                polygonBin, llPolygonABufferSizesPrefixSum.get(),
                llPolygonBBufferSizesPrefixSum.get(), PolygonAIntersectionPresenceFlags.get(),
                PolygonBIntersectionPresenceFlags.get(), isAConst, isBConst, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }
};
