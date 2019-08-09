#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUArithmetic.cuh"
#include "GPUMemory.cuh"
#include "GPUReconstruct.cuh"

#include "../../../cub/cub.cuh"
#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "cuda_ptr.h"


namespace PolygonFunctions
{
struct polyIntersect
{
    __device__ __host__ void operator()(bool turnTable[2]) const
    {
        turnTable[0] = true;
        turnTable[1] = true;
    }
};

struct polyUnion
{
    __device__ __host__ void operator()(bool turnTable[2]) const
    {
        turnTable[0] = false;
        turnTable[1] = false;
    }
};
} // namespace PolygonFunctions

// A point in the linked list of polygons
struct LLPolyVertex
{
    NativeGeoPoint vertex; // The vertex coordinates

    bool isIntersection; // Is this an intersection or a polygon vertex
    bool isValidIntersection; // Is this a valid interection ? ( does the point lie between the crossing lines)
    bool isEntry; // Is this an entry (true) or an exit (false) to the other polygon
    bool wasProcessed; // Was the vertex processed already during the result clipping

    float distanceAlongA; // Distance of the intersection from the beginning of the first line
    float distanceAlongB; // Distance of the intersection from the beginning of the second line

    int32_t prevIdx; // Index of the previous member in the LL
    int32_t nextIdx; // Index of the next member in the LL
    int32_t crossIdx; // Index in the other complex polygon for cross linking during traversal
};

// Calcualte an intersection point between two lines
__device__ LLPolyVertex calc_intersect(NativeGeoPoint sA, NativeGeoPoint eA, NativeGeoPoint sB, NativeGeoPoint eB);

// Calculate the required sizes of the linked lists
// This is clacluated as n + k + intersection_count where n and k
// are the counts of vertices of the complex polygons A and B
__global__ void kernel_calc_LL_buffers_size(int32_t* LLPolygonABufferSizes,
                                            int32_t* LLPolygonBBufferSizes,
                                            GPUMemory::GPUPolygon polygonA,
                                            GPUMemory::GPUPolygon polygonB,
                                            int32_t dataElementCount);

// Build the linked lists from the polygons
__global__ void kernel_build_LL(LLPolyVertex* LLPolygonBuffers,
                                GPUMemory::GPUPolygon polygon,
                                int32_t* LLPolygonBufferSizesPrefixSum,
                                int32_t dataElementCount);

// Insert the intersections into the linked lists and cross link the intersections between complex polygons
__global__ void kernel_add_and_crosslink_intersections_to_LL(LLPolyVertex* LLPolygonABuffers,
                                                             LLPolyVertex* LLPolygonBBuffers,
                                                             GPUMemory::GPUPolygon polygonA,
                                                             GPUMemory::GPUPolygon polygonB,
                                                             int32_t* LLPolygonABufferSizesPrefixSum,
                                                             int32_t* LLPolygonBBufferSizesPrefixSum,
                                                             int32_t dataElementCount);

// Check if a point is withing a complex polygon at a given index
__device__ bool
is_point_in_complex_polygon_at(NativeGeoPoint geoPoint, GPUMemory::GPUPolygon polygon, int32_t idx);

// Decide which intersection points are entry points and whoch ones are exit points and label them accordingly
__global__ void kernel_label_intersections(LLPolyVertex* LLPolygonBuffers,
                                           GPUMemory::GPUPolygon polygonPrimary,
                                           GPUMemory::GPUPolygon polygonSecondary,
                                           int32_t* LLPolygonBufferSizesPrefixSum,
                                           int32_t dataElementCount);

// Clip the polygons in different phases
template <typename OP>
__device__ void clip_polygons(int32_t* polyCount,
                              int32_t* polyIdx,
                              int32_t* pointCount,
                              int32_t* pointIdx,
                              NativeGeoPoint* polyPoints,
                              LLPolyVertex* LLPolygonABuffers,
                              LLPolyVertex* LLPolygonBBuffers,
                              GPUMemory::GPUPolygon polygonA,
                              GPUMemory::GPUPolygon polygonB,
                              int32_t* LLPolygonABufferSizesPrefixSum,
                              int32_t* LLPolygonBBufferSizesPrefixSum,
                              int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Get the complex polygon vertex counts n and k
        int32_t n = GPUMemory::TotalPointCountAt(polygonA, i);
        int32_t k = GPUMemory::TotalPointCountAt(polygonB, i);

        int32_t begIdxA = ((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) + n;
        int32_t endIdxA = LLPolygonABufferSizesPrefixSum[i] - 1;

        int32_t begIdxB = ((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) + k;
        int32_t endIdxB = LLPolygonBBufferSizesPrefixSum[i] - 1;

        // Fill the turn table according to clipping operation
        bool turnTable[2];
        OP{}(turnTable);

        // Calculate the component count
        int32_t componentCount = 0;

        int32_t turnNumber = 0;
        LLPolyVertex* LLPolygonBuffersTable[2] = {LLPolygonABuffers, LLPolygonBBuffers};
        for (int32_t point = begIdxA; point <= endIdxA; point++)
        {
            if (LLPolygonBuffersTable[turnNumber][point].wasProcessed == false)
            {
                // Calculate the sub component count
                int32_t subComponentCount = 0;

                int32_t nextIdx = point;
                do
                {
                    LLPolygonBuffersTable[turnNumber][nextIdx].wasProcessed = true;
                    LLPolygonBuffersTable[1 - turnNumber][LLPolygonBuffersTable[turnNumber][nextIdx].crossIdx].wasProcessed = true;

                    bool forward = (LLPolygonBuffersTable[turnNumber][nextIdx].isEntry == turnTable[turnNumber]);
                    do
                    {
                        // Write the output point
                        if(polyCount && pointCount && polyPoints)
                        {
                            int32_t poly_idx = i;
                            int32_t point_idx = (((poly_idx == 0) ? 0 : polyIdx[poly_idx - 1]) + componentCount);
                            int32_t polyPoint_idx = (((point_idx == 0) ? 0 : pointIdx[point_idx - 1]) + subComponentCount);
                            polyPoints[polyPoint_idx] = LLPolygonBuffersTable[turnNumber][nextIdx].vertex;
                        }

                        if (forward)
                        {
                            nextIdx = LLPolygonBuffersTable[turnNumber][nextIdx].nextIdx;
                        }
                        else
                        {
                            nextIdx = LLPolygonBuffersTable[turnNumber][nextIdx].prevIdx;
                        }

                        subComponentCount++;
                    } while (!LLPolygonBuffersTable[turnNumber][nextIdx].isIntersection);

                    nextIdx = LLPolygonBuffersTable[turnNumber][nextIdx].crossIdx;
                    turnNumber = 1 - turnNumber;
                } while (!LLPolygonBuffersTable[turnNumber][nextIdx].wasProcessed);

                if (polyCount && pointCount)
                {
                    int32_t poly_idx = i;
                    int32_t point_idx = (((poly_idx == 0) ? 0 : polyIdx[poly_idx - 1]) + componentCount);
                    pointCount[point_idx] = subComponentCount;
                }

                componentCount++;
            }
        }

        // Reset the processed flags for the next reconstruction operation
        for (int32_t pointA = begIdxA; pointA <= endIdxA; pointA++)
        {
            LLPolygonABuffers[pointA].wasProcessed = false;
        }

        for (int32_t pointB = begIdxB; pointB <= endIdxB; pointB++)
        {
            LLPolygonBBuffers[pointB].wasProcessed = false;
        }

        // Write the results
        if (polyCount)
        {
            int32_t poly_idx = i;
            polyCount[poly_idx] = componentCount;
        }
    }
}

// A set of methods for clipping
// Reconstruct the polyIdx of the result polygon
template <typename OP>
__global__ void kernel_clip_polyIdx(int32_t* polyCount,
                                    LLPolyVertex* LLPolygonABuffers,
                                    LLPolyVertex* LLPolygonBBuffers,
                                    GPUMemory::GPUPolygon polygonA,
                                    GPUMemory::GPUPolygon polygonB,
                                    int32_t* LLPolygonABufferSizesPrefixSum,
                                    int32_t* LLPolygonBBufferSizesPrefixSum,
                                    int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, nullptr, nullptr, nullptr, nullptr, LLPolygonABuffers,
                      LLPolygonBBuffers, polygonA, polygonB, LLPolygonABufferSizesPrefixSum,
                      LLPolygonBBufferSizesPrefixSum, dataElementCount);
}

// Reconstruct the pointIdx of the result polygon
template <typename OP>
__global__ void kernel_clip_pointIdx(int32_t* polyCount,
                                     int32_t* polyIdx,
                                     int32_t* pointCount,
                                     LLPolyVertex* LLPolygonABuffers,
                                     LLPolyVertex* LLPolygonBBuffers,
                                     GPUMemory::GPUPolygon polygonA,
                                     GPUMemory::GPUPolygon polygonB,
                                     int32_t* LLPolygonABufferSizesPrefixSum,
                                     int32_t* LLPolygonBBufferSizesPrefixSum,
                                     int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, polyIdx, pointCount, nullptr, nullptr, LLPolygonABuffers,
                      LLPolygonBBuffers, polygonA, polygonB, LLPolygonABufferSizesPrefixSum,
                      LLPolygonBBufferSizesPrefixSum, dataElementCount);
}

// Reconstruc the polyPoints of the result polygon
template <typename OP>
__global__ void kernel_clip_polyPoints(int32_t* polyCount,
                                       int32_t* polyIdx,
                                       int32_t* pointCount,
                                       int32_t* pointIdx,
                                       NativeGeoPoint* polyPoints,
                                       LLPolyVertex* LLPolygonABuffers,
                                       LLPolyVertex* LLPolygonBBuffers,
                                       GPUMemory::GPUPolygon polygonA,
                                       GPUMemory::GPUPolygon polygonB,
                                       int32_t* LLPolygonABufferSizesPrefixSum,
                                       int32_t* LLPolygonBBufferSizesPrefixSum,
                                       int32_t dataElementCount)
{
    clip_polygons<OP>(polyCount, polyIdx, pointCount, pointIdx, polyPoints, LLPolygonABuffers,
                      LLPolygonBBuffers, polygonA, polygonB, LLPolygonABufferSizesPrefixSum,
                      LLPolygonBBufferSizesPrefixSum, dataElementCount);
}

class GPUPolygonClipping
{
public:
    // This method expects polygonOut to be with unallocated arrays !!!
    // returns - isEmpty
    template <typename OP>
    static bool ColCol(GPUMemory::GPUPolygon &polygonOut,
                       GPUMemory::GPUPolygon &polygonAin,
                       GPUMemory::GPUPolygon &polygonBin,
                       int32_t dataElementCount)
    {
        // Create buffers for the linked lists
        cuda_ptr<int32_t> LLPolygonABufferSizes(dataElementCount);
        cuda_ptr<int32_t> LLPolygonBBufferSizes(dataElementCount);

        // Calcualte the required buffer sizes
        kernel_calc_LL_buffers_size<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDimPoly()>>>(
            LLPolygonABufferSizes.get(), LLPolygonBBufferSizes.get(), polygonAin, polygonBin, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // DEBUG
        // std::vector<int32_t> LLPolygonABufferSizesDebug(dataElementCount);
        // std::vector<int32_t> LLPolygonBBufferSizesDebug(dataElementCount);

        // GPUMemory::copyDeviceToHost(&LLPolygonABufferSizesDebug[0], LLPolygonABufferSizes.get(), dataElementCount);
        // GPUMemory::copyDeviceToHost(&LLPolygonBBufferSizesDebug[0], LLPolygonBBufferSizes.get(), dataElementCount);

        // for (auto& a : LLPolygonABufferSizesDebug)
        // {
        //     printf("%d ", a);
        // }
        // printf("\n");
        // for (auto& b : LLPolygonBBufferSizesDebug)
        // {
        //     printf("%d ", b);
        // }
        // printf("\n");


        // Calculate the inclusive prefix sum for the LL buffer sizes counters for adressing purpose
        cuda_ptr<int32_t> LLPolygonABufferSizesPrefixSum(dataElementCount);
        cuda_ptr<int32_t> LLPolygonBBufferSizesPrefixSum(dataElementCount);

        GPUReconstruct::PrefixSum(LLPolygonABufferSizesPrefixSum.get(), LLPolygonABufferSizes.get(), dataElementCount);
        GPUReconstruct::PrefixSum(LLPolygonBBufferSizesPrefixSum.get(), LLPolygonBBufferSizes.get(), dataElementCount);

        // Copy back the total size of the LL buffers
        int32_t LLPolygonABufferSizesTotal;
        int32_t LLPolygonBBufferSizesTotal;

        GPUMemory::copyDeviceToHost(&LLPolygonABufferSizesTotal,
                                    LLPolygonABufferSizesPrefixSum.get() + dataElementCount - 1, 1);
        GPUMemory::copyDeviceToHost(&LLPolygonBBufferSizesTotal,
                                    LLPolygonBBufferSizesPrefixSum.get() + dataElementCount - 1, 1);

        // DEBUG
        // std::printf("Sizes A total: %d\n", LLPolygonABufferSizesTotal);
        // std::printf("Sizes B total: %d\n", LLPolygonBBufferSizesTotal);
        // std::printf("\n");

        // Alloc the linked list buffers for the polygon clipping
        cuda_ptr<LLPolyVertex> LLPolygonABuffers(LLPolygonABufferSizesTotal);
        cuda_ptr<LLPolyVertex> LLPolygonBBuffers(LLPolygonBBufferSizesTotal);

        // Transform the complex polygons into linked lists
        // A polygon
        kernel_build_LL<<<Context::getInstance().calcGridDim(dataElementCount),
                          Context::getInstance().getBlockDim()>>>(LLPolygonABuffers.get(), polygonAin,
                                                                  LLPolygonABufferSizesPrefixSum.get(),
                                                                  dataElementCount);
        CheckCudaError(cudaGetLastError());

        // B polygon
        kernel_build_LL<<<Context::getInstance().calcGridDim(dataElementCount),
                          Context::getInstance().getBlockDim()>>>(LLPolygonBBuffers.get(), polygonBin,
                                                                  LLPolygonBBufferSizesPrefixSum.get(),
                                                                  dataElementCount);
        CheckCudaError(cudaGetLastError());

        // DEBUG
        // std::vector<LLPolyVertex> LLa(LLPolygonABufferSizesTotal);
        // std::vector<LLPolyVertex> LLb(LLPolygonBBufferSizesTotal);

        // GPUMemory::copyDeviceToHost(&LLa[0], LLPolygonABuffers.get(), LLPolygonABufferSizesTotal);
        // GPUMemory::copyDeviceToHost(&LLb[0], LLPolygonBBuffers.get(), LLPolygonBBufferSizesTotal);

        // for (int32_t i = 0; i < LLa.size(); i++)
        // {
        //     printf("%2d: %2d %2d\n", i, LLa[i].prevIdx, LLa[i].nextIdx);
        // }
        // printf("\n");
        // for (int32_t i = 0; i < LLb.size(); i++)
        // {
        //     printf("%2d: %2d %2d\n", i, LLb[i].prevIdx, LLb[i].nextIdx);
        // }
        // printf("\n");

        // Insert the intersections into the linked lists and cross link the intersections for intersect/union traversal
        kernel_add_and_crosslink_intersections_to_LL<<<Context::getInstance().calcGridDim(dataElementCount),
                                                       Context::getInstance().getBlockDim()>>>(
            LLPolygonABuffers.get(), LLPolygonBBuffers.get(), polygonAin, polygonBin,
            LLPolygonABufferSizesPrefixSum.get(), LLPolygonBBufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        // DEBUG
        // std::vector<LLPolyVertex> LLa(LLPolygonABufferSizesTotal);
        // std::vector<LLPolyVertex> LLb(LLPolygonBBufferSizesTotal);

        // GPUMemory::copyDeviceToHost(&LLa[0], LLPolygonABuffers.get(), LLPolygonABufferSizesTotal);
        // GPUMemory::copyDeviceToHost(&LLb[0], LLPolygonBBuffers.get(), LLPolygonBBufferSizesTotal);

        // for (int32_t i = 0; i < LLa.size(); i++)
        // {
        //     printf("%2d: %2d %2d %2d\n", i, LLa[i].prevIdx, LLa[i].nextIdx, LLa[i].crossIdx);
        // }
        // printf("\n");
        // for (int32_t i = 0; i < LLb.size(); i++)
        // {
        //     printf("%2d: %2d %2d %2d\n", i, LLb[i].prevIdx, LLb[i].nextIdx, LLb[i].crossIdx);
        // }
        // printf("\n");

        // Decide which intersection points are entry points and whoch ones are exit points
        // and label them accordingly
        kernel_label_intersections<<<Context::getInstance().calcGridDim(dataElementCount),
                                     Context::getInstance().getBlockDim()>>>(
            LLPolygonABuffers.get(), polygonAin, polygonBin, LLPolygonABufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        kernel_label_intersections<<<Context::getInstance().calcGridDim(dataElementCount),
                                     Context::getInstance().getBlockDim()>>>(
            LLPolygonBBuffers.get(), polygonBin, polygonAin, LLPolygonBBufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        // DEBUG
        // std::vector<LLPolyVertex> LLa(LLPolygonABufferSizesTotal);
        // std::vector<LLPolyVertex> LLb(LLPolygonBBufferSizesTotal);

        // GPUMemory::copyDeviceToHost(&LLa[0], LLPolygonABuffers.get(), LLPolygonABufferSizesTotal);
        // GPUMemory::copyDeviceToHost(&LLb[0], LLPolygonBBuffers.get(), LLPolygonBBufferSizesTotal);

        // for (int32_t i = 0; i < LLa.size(); i++)
        // {
        //     printf("%2d: %2d %2d %2d\n", i, LLa[i].prevIdx, LLa[i].nextIdx, LLa[i].isEntry);
        // }
        // printf("\n");
        // for (int32_t i = 0; i < LLb.size(); i++)
        // {
        //     printf("%2d: %2d %2d %2d\n", i, LLb[i].prevIdx, LLb[i].nextIdx, LLb[i].isEntry);
        // }
        // printf("\n");

        // Clip the prepared data structure and produce the result complex polygins
        // polygonOut.polyIdx
        // polygonOut.pointIdx
        // polygonOut.polyPoints

        // Process the polyIdx array
        cuda_ptr<int32_t> polyCount(dataElementCount);

        kernel_clip_polyIdx<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), LLPolygonABuffers.get(), LLPolygonBBuffers.get(), polygonAin, polygonBin,
                LLPolygonABufferSizesPrefixSum.get(), LLPolygonBBufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUMemory::alloc(&(polygonOut.polyIdx), dataElementCount);
        GPUReconstruct::PrefixSum(polygonOut.polyIdx, polyCount.get(), dataElementCount);

        // Retrieve the pointIdx array length
        int32_t pointIdxSize;
        GPUMemory::copyDeviceToHost(&pointIdxSize, polygonOut.polyIdx + dataElementCount - 1, 1);

        // DEBUG
        std::vector<int32_t> polyCount_cpu(dataElementCount);
        std::vector<int32_t> polyIdx_cpu(dataElementCount);

        GPUMemory::copyDeviceToHost(&polyCount_cpu[0], polyCount.get(), dataElementCount);
        GPUMemory::copyDeviceToHost(&polyIdx_cpu[0], polygonOut.polyIdx, dataElementCount);

        for (int32_t i = 0; i < dataElementCount; i++)
        {
            printf("%2d: %2d %2d\n", i, polyCount_cpu[i], polyIdx_cpu[i]);
        }
        printf("\n");

        // Process the pointIdx array
        cuda_ptr<int32_t> pointCount(pointIdxSize);

        kernel_clip_pointIdx<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), polygonOut.polyIdx, pointCount.get(), LLPolygonABuffers.get(),
                LLPolygonBBuffers.get(), polygonAin, polygonBin, LLPolygonABufferSizesPrefixSum.get(),
                LLPolygonBBufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUMemory::alloc(&(polygonOut.pointIdx), pointIdxSize);
        GPUReconstruct::PrefixSum(polygonOut.pointIdx, pointCount.get(), pointIdxSize);

        // Retrieve the polyPoints array length
        int32_t polyPointsSize;
        GPUMemory::copyDeviceToHost(&polyPointsSize, polygonOut.pointIdx + pointIdxSize - 1, 1);

        // DEBUG
        std::vector<int32_t> pointCount_cpu(pointIdxSize);
        std::vector<int32_t> pointIdx_cpu(pointIdxSize);

        GPUMemory::copyDeviceToHost(&pointCount_cpu[0], pointCount.get(), pointIdxSize);
        GPUMemory::copyDeviceToHost(&pointIdx_cpu[0], polygonOut.pointIdx, pointIdxSize);

        for (int32_t i = 0; i < pointIdxSize; i++)
        {
            printf("%2d: %2d %2d\n", i, pointCount_cpu[i], pointIdx_cpu[i]);
        }
        printf("\n");

        // Process the polyPoints array
        GPUMemory::alloc(&(polygonOut.polyPoints), polyPointsSize);

        kernel_clip_polyPoints<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                polyCount.get(), polygonOut.polyIdx, pointCount.get(),polygonOut.pointIdx, polygonOut.polyPoints, LLPolygonABuffers.get(),
                LLPolygonBBuffers.get(), polygonAin, polygonBin, LLPolygonABufferSizesPrefixSum.get(),
                LLPolygonBBufferSizesPrefixSum.get(), dataElementCount);
        CheckCudaError(cudaGetLastError());

        // DEBUG
        std::vector<NativeGeoPoint> polyPoints_cpu(polyPointsSize);

        GPUMemory::copyDeviceToHost(&polyPoints_cpu[0], polygonOut.polyPoints, polyPointsSize);

        for (int32_t i = 0; i < polyPointsSize; i++)
        {
            //printf("%2d: %.2f %.2f\n", i, polyPoints_cpu[i].latitude, polyPoints_cpu[i].longitude);
            printf("[%.2f, %.2f],\n", polyPoints_cpu[i].latitude, polyPoints_cpu[i].longitude);
        }
        printf("\n");

        return true;
    }
};
