#include "GPUPolygonClipping.cuh"

__device__ LLPolyVertex calc_intersect(NativeGeoPoint sA, NativeGeoPoint eA, NativeGeoPoint sB, NativeGeoPoint eB)
{
    float adx = eA.latitude - sA.latitude;
    float ady = eA.longitude - sA.longitude;
    float bdx = eB.latitude - sB.latitude;
    float bdy = eB.longitude - sB.longitude;

    float axb = adx * bdy - ady * bdx;

    if (axb == 0)
    {
        LLPolyVertex retFail = {{0, 0}, 0x0, -1.0, -1.0, -1, -1, -1};

        setIsIntersection(retFail, true);
        setIsValidIntersection(retFail, false);
        setIsEntry(retFail, false);
        setWasProcessed(retFail, false);

        return retFail;
    }

    float dx = sA.latitude - sB.latitude;
    float dy = sA.longitude - sB.longitude;

    float alongA = (bdx * dy - bdy * dx) / axb;
    float alongB = (adx * dy - ady * dx) / axb;

    bool intersectionValidity = (alongA > 0 && alongA < 1 && alongB > 0 && alongB < 1);

    LLPolyVertex ret = {{sA.latitude + alongA * adx, sA.longitude + alongA * ady},
                        0x0,
                        alongA,
                        alongB,
                        -1,
                        -1,
                        -1};

	setHasIntersections(ret, intersectionValidity);
    setIsIntersection(ret, true);
    setIsValidIntersection(ret, intersectionValidity);
    setIsEntry(ret, false);
    setWasProcessed(ret, false);

    return ret;
}

__global__ void kernel_calc_LL_buffers_size(int32_t* LLPolygonABufferSizes,
                                            int32_t* LLPolygonBBufferSizes,
                                            int8_t* PolygonAIntersectionPresenceFlags,
                                            int8_t* PolygonBIntersectionPresenceFlags,
                                            GPUMemory::GPUPolygon polygonA,
                                            GPUMemory::GPUPolygon polygonB,
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

        int32_t polyIdxA = GPUMemory::PolyIdxAt(polygonA, iAIdx);
        int32_t polyCountA = GPUMemory::PolyCountAt(polygonA, iAIdx);

        int32_t polyIdxB = GPUMemory::PolyIdxAt(polygonB, iBIdx);
        int32_t polyCountB = GPUMemory::PolyCountAt(polygonB, iBIdx);

        int32_t intersectCount = 0;
        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            int32_t pointIdxA = GPUMemory::PointIdxAt(polygonA, a);
            int32_t pointCountA = GPUMemory::PointCountAt(polygonA, a);

            int8_t intersectionPresentInSubPolygonA = 0;

            for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                int32_t pointIdxB = GPUMemory::PointIdxAt(polygonB, b);
                int32_t pointCountB = GPUMemory::PointCountAt(polygonB, b);

                int8_t intersectionPresentInSubPolygonB = 0;

                // Calculate total intersections count
                for (int32_t pointA = pointIdxA; pointA < (pointIdxA + pointCountA); pointA++)
                {
                    for (int32_t pointB = pointIdxB; pointB < (pointIdxB + pointCountB); pointB++)
                    {
                        LLPolyVertex intersection =
                            calc_intersect(polygonA.polyPoints[pointA],
                                           polygonA.polyPoints[pointIdxA + (pointA - pointIdxA + 1) % pointCountA],
                                           polygonB.polyPoints[pointB],
                                           polygonB.polyPoints[pointIdxB + (pointB - pointIdxB + 1) % pointCountB]);

                        if (getIsValidIntersection(intersection))
                        {
                            intersectionPresentInSubPolygonA = 1;
                            intersectionPresentInSubPolygonB = 1;

                            intersectCount++;
                        }
                    }
                }
                PolygonBIntersectionPresenceFlags[isBConst ? b + i * dataElementCount : b] |=
                    intersectionPresentInSubPolygonB;
            }
            PolygonAIntersectionPresenceFlags[isAConst ? a + i * dataElementCount : a] |=
                intersectionPresentInSubPolygonA;
        }

        // Get the complex polygon vertex counts n and k
        int32_t n = GPUMemory::TotalPointCountAt(polygonA, iAIdx);
        int32_t k = GPUMemory::TotalPointCountAt(polygonB, iBIdx);

        // Assign the calculated buffers size
        LLPolygonABufferSizes[i] = n + intersectCount;
        LLPolygonBBufferSizes[i] = k + intersectCount;
    }
}

__global__ void kernel_build_LL(LLPolyVertex* LLPolygonBuffers,
                                GPUMemory::GPUPolygon polygon,
                                int32_t* LLPolygonBufferSizesPrefixSum,
                                int8_t* PolygonIntersectionPresenceFlags,
                                bool isConst,
                                int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t iIdx = isConst ? 0 : i;

        int32_t LLPolygonEndIdx = 0;

        int32_t polyIdx = GPUMemory::PolyIdxAt(polygon, iIdx);
        int32_t polyCount = GPUMemory::PolyCountAt(polygon, iIdx);

        // Transform polygon
        for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
        {
            int32_t pointIdx = GPUMemory::PointIdxAt(polygon, p);
            int32_t pointCount = GPUMemory::PointCountAt(polygon, p);

            for (int32_t point = pointIdx; point < (pointIdx + pointCount); point++)
            {
                int32_t localIdx = pointIdx - GPUMemory::PointIdxAt(polygon, polyIdx);

                // Set the linked list entry
                LLPolyVertex vertex = {
                    polygon.polyPoints[point],
                    0x0,
                    -1.0,
                    -1.0,
                    ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx +
                        (point - pointIdx - 1 + pointCount) % pointCount,
                    ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx +
                        (point - pointIdx + 1) % pointCount,
                    -1};

				setHasIntersections(vertex, PolygonIntersectionPresenceFlags[p]);
                setIsIntersection(vertex, false);
                setIsValidIntersection(vertex, false);
                setIsEntry(vertex, false);
                setWasProcessed(vertex, false);

                LLPolygonBuffers[((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + LLPolygonEndIdx] = vertex;

                // Increment the local pointer to the end of the LL
                LLPolygonEndIdx++;
            }
        }
    }
}

__global__ void kernel_add_and_crosslink_intersections_to_LL(LLPolyVertex* LLPolygonABuffers,
                                                             LLPolyVertex* LLPolygonBBuffers,
                                                             GPUMemory::GPUPolygon polygonA,
                                                             GPUMemory::GPUPolygon polygonB,
                                                             int32_t* LLPolygonABufferSizesPrefixSum,
                                                             int32_t* LLPolygonBBufferSizesPrefixSum,
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

        // "Pointers" to the element after the last valid element of the linked lists
        // They begin after the last non intersection e.g. poly vertex address
        int32_t LLPolygonAEndIdx = GPUMemory::TotalPointCountAt(polygonA, iAIdx);
        int32_t LLPolygonBEndIdx = GPUMemory::TotalPointCountAt(polygonB, iBIdx);

        int32_t polyIdxA = GPUMemory::PolyIdxAt(polygonA, iAIdx);
        int32_t polyCountA = GPUMemory::PolyCountAt(polygonA, iAIdx);

        int32_t polyIdxB = GPUMemory::PolyIdxAt(polygonB, iBIdx);
        int32_t polyCountB = GPUMemory::PolyCountAt(polygonB, iBIdx);

        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            int32_t pointIdxA = GPUMemory::PointIdxAt(polygonA, a);
            int32_t pointCountA = GPUMemory::PointCountAt(polygonA, a);

            for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                int32_t pointIdxB = GPUMemory::PointIdxAt(polygonB, b);
                int32_t pointCountB = GPUMemory::PointCountAt(polygonB, b);

                // Calculate intersections and insert them into the LL
                for (int32_t pointA = pointIdxA; pointA < (pointIdxA + pointCountA); pointA++)
                {
                    for (int32_t pointB = pointIdxB; pointB < (pointIdxB + pointCountB); pointB++)
                    {
                        LLPolyVertex intersection =
                            calc_intersect(polygonA.polyPoints[pointA],
                                           polygonA.polyPoints[pointIdxA + (pointA - pointIdxA + 1) % pointCountA],
                                           polygonB.polyPoints[pointB],
                                           polygonB.polyPoints[pointIdxB + (pointB - pointIdxB + 1) % pointCountB]);

                        // If an intersection is valid, insert it into the linked lists and create a cross reference
                        if (getIsValidIntersection(intersection))
                        {
                            int32_t LLPolygonAEndIdxLocal =
                                ((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) + LLPolygonAEndIdx;
                            int32_t LLPolygonBEndIdxLocal =
                                ((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) + LLPolygonBEndIdx;

                            // Save the intersection data
                            LLPolygonABuffers[LLPolygonAEndIdxLocal] = intersection;
                            LLPolygonBBuffers[LLPolygonBEndIdxLocal] = intersection;

                            // Write the cross reference indices
                            LLPolygonABuffers[LLPolygonAEndIdxLocal].crossIdx = LLPolygonBEndIdxLocal;
                            LLPolygonBBuffers[LLPolygonBEndIdxLocal].crossIdx = LLPolygonAEndIdxLocal;

                            // "Rewire" the prev and next pointers in both linked lists
                            // so that the point is in it's correct place
                            // according to the parametric distance from the beginning of the line segment
                            //////////////////////////////////////////////////////////////////////////////
                            // First polygon - A
                            int32_t localIdxA = pointIdxA - GPUMemory::PointIdxAt(polygonA, polyIdxA);

                            int32_t begIdxA = ((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) +
                                              localIdxA + (pointA - pointIdxA) % pointCountA;
                            int32_t endIdxA = ((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) +
                                              localIdxA + (pointA - pointIdxA + 1) % pointCountA;

                            int32_t nextIdxA = LLPolygonABuffers[begIdxA].nextIdx;
                            while (nextIdxA != endIdxA && LLPolygonABuffers[LLPolygonAEndIdxLocal].distanceAlongA >
                                                              LLPolygonABuffers[nextIdxA].distanceAlongA)
                            {
                                nextIdxA = LLPolygonABuffers[nextIdxA].nextIdx;
                            }

                            // Rewire the pointers for the first polygon - A
                            LLPolygonABuffers[LLPolygonAEndIdxLocal].prevIdx =
                                LLPolygonABuffers[nextIdxA].prevIdx;
                            LLPolygonABuffers[LLPolygonAEndIdxLocal].nextIdx = nextIdxA;

                            LLPolygonABuffers[LLPolygonABuffers[nextIdxA].prevIdx].nextIdx = LLPolygonAEndIdxLocal;
                            LLPolygonABuffers[nextIdxA].prevIdx = LLPolygonAEndIdxLocal;
                            //////////////////////////////////////////////////////////////////////////////
                            // Second polygon - B
                            int32_t localIdxB = pointIdxB - GPUMemory::PointIdxAt(polygonB, polyIdxB);

                            int32_t begIdxB = ((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) +
                                              localIdxB + (pointB - pointIdxB) % pointCountB;
                            int32_t endIdxB = ((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) +
                                              localIdxB + (pointB - pointIdxB + 1) % pointCountB;

                            int32_t nextIdxB = LLPolygonBBuffers[begIdxB].nextIdx;
                            while (nextIdxB != endIdxB && LLPolygonBBuffers[LLPolygonBEndIdxLocal].distanceAlongB >
                                                              LLPolygonBBuffers[nextIdxB].distanceAlongB)
                            {
                                nextIdxB = LLPolygonBBuffers[nextIdxB].nextIdx;
                            }

                            // Rewire the pointers for the second polygon - B
                            LLPolygonBBuffers[LLPolygonBEndIdxLocal].prevIdx =
                                LLPolygonBBuffers[nextIdxB].prevIdx;
                            LLPolygonBBuffers[LLPolygonBEndIdxLocal].nextIdx = nextIdxB;

                            LLPolygonBBuffers[LLPolygonBBuffers[nextIdxB].prevIdx].nextIdx = LLPolygonBEndIdxLocal;
                            LLPolygonBBuffers[nextIdxB].prevIdx = LLPolygonBEndIdxLocal;
                            //////////////////////////////////////////////////////////////////////////////

                            // Increment the LL end pointers
                            LLPolygonAEndIdx++;
                            LLPolygonBEndIdx++;
                        }
                    }
                }
            }
        }
    }
}

__device__ bool is_point_in_complex_polygon_at(NativeGeoPoint geoPoint, GPUMemory::GPUPolygon polygon, int32_t idx)
{
    bool isPointInPolygon = false;

    int32_t polyIdx = GPUMemory::PolyIdxAt(polygon, idx);
    int32_t polyCount = GPUMemory::PolyCountAt(polygon, idx);

    for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
    {
        int32_t pointIdx = GPUMemory::PointIdxAt(polygon, p);
        int32_t pointCount = GPUMemory::PointCountAt(polygon, p);

        // Dank raycasting magic as seen in GPUPolygonContains
        for (int32_t point = pointIdx; point < (pointIdx + pointCount); point++)
        {
            int32_t pBeg = point;
            int32_t pEnd = pointIdx + (point - pointIdx + 1) % pointCount;

            if (((polygon.polyPoints[pBeg].longitude > geoPoint.longitude) !=
                 (polygon.polyPoints[pEnd].longitude > geoPoint.longitude)) &&
                (geoPoint.latitude <
                 (polygon.polyPoints[pEnd].latitude - polygon.polyPoints[pBeg].latitude) *
                         (geoPoint.longitude - polygon.polyPoints[pBeg].longitude) /
                         (polygon.polyPoints[pEnd].longitude - polygon.polyPoints[pBeg].longitude) +
                     polygon.polyPoints[pBeg].latitude))
            {
                isPointInPolygon = !isPointInPolygon;
            }
        }
    }

    return isPointInPolygon;
}

__global__ void kernel_label_intersections(LLPolyVertex* LLPolygonBuffers,
                                           GPUMemory::GPUPolygon polygonPrimary,
                                           GPUMemory::GPUPolygon polygonSecondary,
                                           int32_t* LLPolygonBufferSizesPrefixSum,
                                           bool isPrimaryConst,
                                           bool isSecondaryConst,
                                           int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t iPrimary = isPrimaryConst ? 0 : i;
        int32_t iSecondary = isSecondaryConst ? 0 : i;

        int32_t polyIdx = GPUMemory::PolyIdxAt(polygonPrimary, iPrimary);
        int32_t polyCount = GPUMemory::PolyCountAt(polygonPrimary, iPrimary);

        for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
        {
            int32_t pointIdx = GPUMemory::PointIdxAt(polygonPrimary, p);
            int32_t pointCount = GPUMemory::PointCountAt(polygonPrimary, p);

            // Iterate trough the linked list for the current sub polygon and label the intersections
            int32_t localIdx = pointIdx - GPUMemory::PointIdxAt(polygonPrimary, polyIdx);

            int32_t begIdx = ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx;
            int32_t endIdx = ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx + pointCount - 1;

            // Check the inclusion of the first point in the other polygon
            bool isPointInPolygon =
                !is_point_in_complex_polygon_at(LLPolygonBuffers[begIdx].vertex, polygonSecondary, iSecondary);

            int32_t nextIdx = begIdx;
            do
            {
                // If the given vertex is an intersection - assign the correct entry/exit label
                if (getIsIntersection(LLPolygonBuffers[nextIdx]))
                {
                    setIsEntry(LLPolygonBuffers[nextIdx], isPointInPolygon);
                    isPointInPolygon = !isPointInPolygon;
                }

                nextIdx = LLPolygonBuffers[nextIdx].nextIdx;
            } while (nextIdx != begIdx);
        }
    }
}