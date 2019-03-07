#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUComplexPolygon.cuh"

#include "../../NativeGeoPoint.h"
#include "../Context.h"

/// <summary>
/// Check whether point is in given polygon using GPU
/// </summary>
/// <param name="outMask">pointer to output mask</param>
/// <param name="geoPointsInput">points to check for inclusion</param>
/// <param name="geoPoints">points of all polygons</param>
/// <param name="complexPolygonIdx">Start indices of range of polygons in polygon arrays, for each complex polygon</param>
/// <param name="complexPolygonCnt">Length of the polygon range for each complex polygon</param>
/// <param name="polygonIdx">Start indices of range of points in points array, for each polygon</param>
/// <param name="polygonCnt">Length of the point range for each polygon</param>
/// <param name="pointCount">Length of geoPointsInput</param>
/// <param name="polygonCount">Length of complexPolygonIdx and complexPolygonCnt</param>
/// <remarks>If point count is equal to 1, the point is checked against every polygon.
/// If polygon count is equal to 1, the polygon is checked against every point.
/// If point count is equal to polygon count, points are checked one to one against polygons on the same array index.
/// </remarks>
__global__ void
kernel_point_in_polygon(int8_t* outMask, ComplexPolygon polygonCol, NativeGeoPoint* geoPointsInput, int32_t pointCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx;
         i < (pointCount > polygonCol.polygonCount ? pointCount : polygonCol.polygonCount); i += stride)
    {
        NativeGeoPoint point;

        if (pointCount == 1)
        {
            point = geoPointsInput[0];
        }
        else
        {
            point = geoPointsInput[i];
        }

        int32_t polyIdx = i;

        if (polygonCol.polygonCount == 1)
        {
            polyIdx = 0;
        }
        int32_t subPolygonsStartIdx = polygonCol.complexPolygonIdx[polyIdx];
        int32_t subPolygonsCount = polygonCol.complexPolygonCnt[polyIdx];
        int8_t result = 0;
        for (int32_t j = subPolygonsStartIdx; j < subPolygonsStartIdx + subPolygonsCount; j++)
        {
            int32_t verticesStartIdx = polygonCol.polygonIdx[j];
            int32_t verticesCount = polygonCol.polygonCnt[j];
            NativeGeoPoint previousVertex = polygonCol.geoPoints[verticesStartIdx];
            NativeGeoPoint currentVertex;
            for (int32_t k = verticesStartIdx + 1; k < verticesStartIdx + verticesCount; k++)
            {
                currentVertex = polygonCol.geoPoints[k];
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

class GPUPolygonContains
{
public:
    /// <summary>
    /// Check whether point is in given polygon
    /// </summary>
    /// <param name="outMask">pointer to output mask</param>
    /// <param name="polygonCol">A structure to represent a complex polygon column</param>
    /// <param name="geoPointsInput">points to check for inclusion</param>
	/// <param name="pointCount">Length of geoPointsInput</param> <param
    /// name="polygonCount">Length of complexPolygonIdx and complexPolygonCnt</param>
    /// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS) or some error
    /// occured (GPU_EXTENSION_ERROR)</returns> <remarks>If point count is equal to 1, the point is
    /// checked against every polygon. If polygon count is equal to 1, the polygon is checked
    /// against every point. If point count is equal to polygon count, points are checked one to one
    /// against polygons on the same array index.
    /// </remarks>
    static void contains(int8_t* outMask, ComplexPolygon polygonCol, NativeGeoPoint* geoPointsInput int32_t pointCount)
    {
        Context& context = Context::getInstance();

        if (pointCount != polygonCount && pointCount != 1 && polygonCount != 1)
        {
            QueryEngineError::setType(QueryEngineError::GPU_EXTENSION_ERROR);
            return;
        }

        kernel_point_in_polygon<<<context.calcGridDim((pointCount > polygonCount ? pointCount : polygonCount)),
                                  context.getBlockDim()>>>(outMask, polygonCol, geoPointsInput, pointCount);

        QueryEngineError::setCudaError(cudaGetLastError());
    }
    /// <summary>
    /// Check whether point is in given polygon
    /// </summary>
    /// <param name="outMask">pointer to output mask</param>
    /// <param name="polygon">A structure to represent a complex polygon</param>
    /// <param name="geoPointsInput">points to check for inclusion</param>
    /// <param name="retSize">requested return size</param>
    /// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS) or some error
    /// occured (GPU_EXTENSION_ERROR)</returns> <remarks>If point count is equal to 1, the point is
    /// checked against every polygon. If polygon count is equal to 1, the polygon is checked
    /// against every point. If point count is equal to polygon count, points are checked one to one
    /// against polygons on the same array index.
    /// </remarks>
    static void containsConst(int8_t* outMask, ComplexPolygon polygon, NativeGeoPoint* geoPointsInput, int32_t retSize)
    {
        Context& context = Context::getInstance();

        kernel_point_in_polygon << <context.calcGridDim(1), context.getBlockDim()>>>(
            (outMask, polygon, geoPointsInput, 1);
		int8_t result;
		GPUMemory::copyDeviceToHost(&result, outMask, 1);
		GPUMemory::memset(outMask, result, retSize);
		QueryEngineError::setCudaError(cudaGetLastError());
    }
};
