#include "GPUPolygonyContains.cuh"

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
	NativeGeoPoint* geoPointCol, int32_t pointCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx;
         i < (pointCount > polygonCount ? pointCount : polygonCount); i += stride)
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

        outMask[i] = point_in_polygon(polyIdx,polygonCol,point);
    }
}

__device__ int8_t point_in_polygon(int polyIdx, GPUMemory::GPUPolygon polygonCol, NativeGeoPoint point)
{
    int8_t result = 0;
    int32_t subPolygonsStartIdx = polygonCol.polyIdx[polyIdx];
    int32_t subPolygonsCount = polygonCol.polyCount[polyIdx];
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
    return result;
}