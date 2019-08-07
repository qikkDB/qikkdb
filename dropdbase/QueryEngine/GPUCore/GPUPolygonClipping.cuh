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

// A point in the linked list of polygons
struct LLPolyVertex
{
    NativeGeoPoint vertex; // The vertex coordinates

    bool isIntersection; // Is this an intersection or a polygon vertex
    bool isValidIntersection; // Is this a valid interection ? ( does the point lie between the crossing lines)
    bool isEntry; // Is this an entry (true) or an exit (false) to the other polygon

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
__device__ bool isPointInComplexPolygonAt(NativeGeoPoint geoPoint, GPUMemory::GPUPolygon polygon, int32_t idx);

// Decide which intersection points are entry points and whoch ones are exit points and label them accordingly
__global__ void kernel_label_intersections(LLPolyVertex* LLPolygonBuffers,
                                           GPUMemory::GPUPolygon polygonPrimary,
                                           GPUMemory::GPUPolygon polygonSecondary,
                                           int32_t* LLPolygonBufferSizesPrefixSum,
                                           int32_t dataElementCount);

class GPUPolygonClipping
{
public:
    template <typename OP>
    static void ColCol(GPUMemory::GPUPolygon polygonOut,
                       GPUMemory::GPUPolygon polygonAin,
                       GPUMemory::GPUPolygon polygonBin,
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
    }
};
