#ifndef GPU_POLYGON_CUH
#define GPU_POLYGON_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "../../NativeGeoPoint.h"

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
__global__
void kernel_point_in_polygon(int8_t *outMask,
	NativeGeoPoint *geoPointsInput,
	NativeGeoPoint *geoPoints,
	int32_t *complexPolygonIdx,
	int32_t *complexPolygonCnt,
	int32_t *polygonIdx,
	int32_t *polygonCnt,
	int32_t pointCount,
	int32_t polygonCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < (pointCount > polygonCount ? pointCount : polygonCount); i += stride)
	{
		int32_t polyIdx = i;
		NativeGeoPoint point;

		if (pointCount == 1)
		{
			point = geoPointsInput[0];
			polyIdx = 0;
		}
		else
		{
			point = geoPointsInput[i];
		}

		int32_t subPolygonsStartIdx = complexPolygonIdx[polyIdx];
		int32_t subPolygonsCount = complexPolygonCnt[polyIdx];
		int8_t result = 0;
		for (int32_t j = subPolygonsStartIdx; j < subPolygonsStartIdx + subPolygonsCount; j++)
		{
			int32_t verticesStartIdx = polygonIdx[j];
			int32_t verticesCount = polygonCnt[j];
			NativeGeoPoint previousVertex = geoPoints[verticesStartIdx];
			NativeGeoPoint currentVertex;
			for (int32_t k = verticesStartIdx + 1; k < verticesStartIdx + verticesCount; k++)
			{
				currentVertex = geoPoints[k];
				// Dark raycasting magic
				if (((currentVertex.latitude > point.latitude) != (previousVertex.latitude > point.latitude)) &&
					(point.longitude < (previousVertex.longitude - currentVertex.longitude) *
					(point.latitude - currentVertex.latitude) / (previousVertex.latitude - currentVertex.latitude)
						+ currentVertex.longitude))
				{
					result = !result;
				}
				previousVertex = currentVertex;
			}
		}
		outMask[i] = result;
	}
}

class GPUPolygon {
public:
	/// <summary>
	/// Check whether point is in given polygon
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
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	/// <remarks>If point count is equal to 1, the point is checked against every polygon.
	/// If polygon count is equal to 1, the polygon is checked against every point.
	/// If point count is equal to polygon count, points are checked one to one against polygons on the same array index.
	/// </remarks>
	static void contains(int8_t *outMask,
		NativeGeoPoint *geoPointsInput,
		NativeGeoPoint *geoPoints,
		int32_t *complexPolygonIdx,
		int32_t *complexPolygonCnt,
		int32_t *polygonIdx,
		int32_t *polygonCnt,
		int32_t pointCount,
		int32_t polygonCount)
	{
		Context& context = Context::getInstance();

		if (pointCount != polygonCount && pointCount != 1 && polygonCount != 1)
		{
			context.getLastError().setType(QueryEngineError::GPU_EXTENSION_ERROR);
			return;
		}

		kernel_point_in_polygon << < context.calcGridDim((pointCount > polygonCount ? pointCount : polygonCount)), context.getBlockDim() >> >
			(outMask, geoPointsInput, geoPoints, complexPolygonIdx, complexPolygonCnt, polygonIdx,
				polygonCnt, pointCount, polygonCount);
		cudaDeviceSynchronize();

		context.getLastError().setCudaError(cudaGetLastError());
	}
};

#endif 

