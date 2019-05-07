#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"
#include "../GPUError.h"

#include "../../NativeGeoPoint.h"
#include "../Context.h"

/// <summary>
/// Check whether point is in given polygon using GPU
/// </summary>
/// <param name="outMask">pointer to output mask</param>
/// <param name="geoPointCol">points to check for inclusion</param>
/// <param name="pointCount">Length of geoPointCol</param>
/// <param name="polygonCount">Length of complexPolygonIdx and complexPolygonCnt</param>
/// <remarks>If point count is equal to 1, the point is checked against every polygon.
/// If polygon count is equal to 1, the polygon is checked against every point.
/// If point count is equal to polygon count, points are checked one to one against polygons on the same array index.
/// </remarks>
__global__ void
kernel_point_in_polygon(int8_t* outMask, GPUMemory::GPUPolygon polygonCol, int32_t polygonCount,
	NativeGeoPoint* geoPointCol, int32_t pointCount);

__device__ int8_t point_in_polygon(int polyIdx, GPUMemory::GPUPolygon polygonCol, NativeGeoPoint point);

class GPUPolygonContains
{
public:
    /// <summary>
    /// Check whether point is in given polygon
    /// </summary>
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
    static void contains(int8_t* outMask, GPUMemory::GPUPolygon polygonCol, int32_t polygonCount,
		NativeGeoPoint* geoPointCol, int32_t pointCount)
    {
        Context& context = Context::getInstance();

        if (pointCount != polygonCount && pointCount != 1 && polygonCount != 1)
        {
            CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR);
            return;
        }

        kernel_point_in_polygon<<<context.calcGridDim((pointCount > polygonCount ? pointCount : polygonCount)),
                                  context.getBlockDim()>>>(outMask, polygonCol, polygonCount, geoPointCol, pointCount);

        CheckCudaError(cudaGetLastError());
    }

    /// <summary>
    /// Check whether point is in given polygon
    /// </summary>
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

        kernel_point_in_polygon << <context.calcGridDim(1), context.getBlockDim()>>>
            (outMask, polygonCol, 1, geoPointCol, 1);
		int8_t result;
		GPUMemory::copyDeviceToHost(&result, outMask, 1);
		GPUMemory::memset(outMask, result, retSize);
		CheckCudaError(cudaGetLastError());
    }
};
