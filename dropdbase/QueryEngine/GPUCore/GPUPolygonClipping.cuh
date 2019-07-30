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
__device__ struct LLPolyVertex
{
    NativeGeoPoint vertex;      // The vertex coordinates

    bool isIntersection;        // Is this an intersection or a polygon vertex
    bool isValidIntersection;   // Is this a valid interection ? ( does the point lie between the crossing lines)

    float distanceAlongA;       // Distance of the intersection from the beginning of the first line
    float distanceAlongB;       // Distance of the intersection from the beginning of the second line

    int32_t prevIdx;            // Index of the previous member in the LL
    int32_t nextIdx;            // Index of the next member in the LL
    int32_t crossIdx;           // Index in the other complex polygon for cross linking during traversal
};

// Calcualte an intersection point between two lines
__device__ LLPolyVertex calc_intersect(NativeGeoPoint sA, NativeGeoPoint eA, 
                                       NativeGeoPoint sB, NativeGeoPoint eB);

// Calculate the required sizes of the linked lists
// This is clacluated as n + k + intersection_count where n and k 
// are the counts of vertices of the complex polygons A and B
__global__ void kernel_calc_LL_buffers_size(int32_t *intesection_counts, 
                                            GPUMemory::GPUPolygon polygonA,
                                            GPUMemory::GPUPolygon polygonB,
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
        // Create a buffer with the sizes of the LL buffers and calcualte the required sizes
        cuda_ptr<int32_t> LLBufferSizes(dataElementCount);
        cuda_ptr<int32_t> LLBufferSizesPrefixSum(dataElementCount);
        kernel_calc_LL_buffers_size<<< Context::getInstance().calcGridDim(dataElementCount), 
                                       Context::getInstance().getBlockDimPoly()>>>(LLBufferSizes.get(), 
                                                                                   polygonAin,
                                                                                   polygonBin,
                                                                                   dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Calculate the inclusive prefix sum for the LL buffer sizes counters for adressing purpose
        GPUReconstruct::PrefixSum(LLBufferSizesPrefixSum.get(), LLBufferSizes.get(), dataElementCount);

        // Copy back the result total size
        int32_t totalLLBuffersSize;
        GPUMemory::copyDeviceToHost(&totalLLBuffersSize, LLBufferSizesPrefixSum.get() + dataElementCount - 1, 1);
        
        std::printf("LL buffer sizes total count: %d\n", totalLLBuffersSize);
        
    }
};
