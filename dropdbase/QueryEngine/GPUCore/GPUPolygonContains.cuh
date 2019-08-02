#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"

#include "../../NativeGeoPoint.h"
#include "../Context.h"

/// Kernel for checking whether point is in given polygon.
/// This is performed in parallel per row, so 1 point is checked with 1 complex polygon,
/// but supported are also versions 1 : n and n : 1.
/// <param name="outMask">pointer to output mask</param>
/// <param name="polygonCol">column with complex polygons (structure with GPU pointers to start of arrays)</param>
/// <param name="polygonCount">number of complex polygons</param>
/// <param name="geoPointCol">column with pointsto check for inclusion</param>
/// <param name="pointCount">number of points</param>
/// <remarks>If point count is equal to 1, the point is checked against every polygon.
/// If polygon count is equal to 1, the polygon is checked against every point.
/// If point count is equal to polygon count, points are checked one to one against polygons on the same array index.
/// </remarks>
__global__ void kernel_point_in_polygon(int8_t* outMask,
                                        GPUMemory::GPUPolygon polygonCol,
                                        int32_t polygonCount,
                                        NativeGeoPoint* geoPointCol,
                                        int32_t pointCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < (pointCount > polygonCount ? pointCount : polygonCount); i += stride)
    {
        NativeGeoPoint point;

        if (pointCount == 1)
        {
            point = geoPointCol[0];
        }
        else
        {
            point = geoPointCol[i];
        }

        int32_t polyIdx = i;

        if (polygonCount == 1)
        {
            polyIdx = 0;
        }
        int32_t subPolygonsStartIdx = polygonCol.polyIdx[polyIdx];
        int32_t subPolygonsCount = polygonCol.polyCount[polyIdx];
        int8_t result = 0;
        for (int32_t j = subPolygonsStartIdx; j < subPolygonsStartIdx + subPolygonsCount; j++)
        {
            int32_t verticesStartIdx = polygonCol.pointIdx[j];
            int32_t verticesCount = polygonCol.pointCount[j];
            NativeGeoPoint previousVertex = polygonCol.polyPoints[verticesStartIdx];
            NativeGeoPoint currentVertex;
            for (int32_t k = verticesStartIdx + 1; k < verticesStartIdx + verticesCount; k++)
            {
                currentVertex = polygonCol.polyPoints[k];
                // Dark raycasting magic
                if (((currentVertex.latitude > point.latitude) != (previousVertex.latitude > point.latitude)) &&
                    (point.longitude < (previousVertex.longitude - currentVertex.longitude) *
                                               (point.latitude - currentVertex.latitude) /
                                               (previousVertex.latitude - currentVertex.latitude) +
                                           currentVertex.longitude))
                {
                    result = !result;
                }
                previousVertex = currentVertex;
            }
        }
        outMask[i] = result;
    }
}

/// Class for checking whether points are in given polygons
class GPUPolygonContains
{
public:
    /// Check whether point is in given polygon (versions n:n, 1:n, n:1)
    /// <param name="outMask">pointer to output mask</param>
    /// <param name="polygonCol">A structure to represent a complex polygon column</param>
    /// <param name="geoPointCol">points to check for inclusion</param>
    /// <param name="pointCount">Length of geoPointCol</param>
    /// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS) or some error
    /// occured (GPU_EXTENSION_ERROR)</returns> <remarks>If point count is equal to 1, the point is
    /// checked against every polygon. If polygon count is equal to 1, the polygon is checked
    /// against every point. If point count is equal to polygon count, points are checked one to one
    /// against polygons on the same array index.
    /// </remarks>
    static void
    contains(int8_t* outMask, GPUMemory::GPUPolygon polygonCol, int32_t polygonCount, NativeGeoPoint* geoPointCol, int32_t pointCount)
    {
        Context& context = Context::getInstance();

        if (pointCount != polygonCount && pointCount != 1 && polygonCount != 1)
        {
            CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                                  "PointCount=" + std::to_string(pointCount) + ", PolygonCount=" +
                                      std::to_string(polygonCount) + ": not allowed combination");
            return;
        }

        kernel_point_in_polygon<<<context.calcGridDim((pointCount > polygonCount ? pointCount : polygonCount)),
                                  context.getBlockDim()>>>(outMask, polygonCol, polygonCount,
                                                           geoPointCol, pointCount);

        CheckCudaError(cudaGetLastError());
    }

    /// Check whether point is in given polygon (const version - 1 : 1)
    /// <param name="outMask">pointer to output mask</param>
    /// <param name="polygonCol">A structure to represent a complex polygon</param>
    /// <param name="geoPointCol">points to check for inclusion</param>
    /// <param name="retSize">requested return size</param>
    /// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS) or some error
    /// occured (GPU_EXTENSION_ERROR)</returns> <remarks>If point count is equal to 1, the point is
    /// checked against every polygon. If polygon count is equal to 1, the polygon is checked
    /// against every point. If point count is equal to polygon count, points are checked one to one
    /// against polygons on the same array index.
    /// </remarks>
    static void containsConst(int8_t* outMask, GPUMemory::GPUPolygon polygonCol, NativeGeoPoint* geoPointCol, int32_t retSize)
    {
        Context& context = Context::getInstance();

        kernel_point_in_polygon<<<context.calcGridDim(1), context.getBlockDim()>>>(outMask, polygonCol,
                                                                                   1, geoPointCol, 1);
        int8_t result;
        GPUMemory::copyDeviceToHost(&result, outMask, 1);
        GPUMemory::memset(outMask, result, retSize);

        CheckCudaError(cudaGetLastError());
    }
};
