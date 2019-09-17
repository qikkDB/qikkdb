#include "GPUPolygonClipping.cuh"

// Bit getters
 __device__ bool LLPolyVertex::GetHasIntersections()
{
    return static_cast<bool>((llflags >> 4) & 0x01);
}

 __device__ bool LLPolyVertex::GetIsIntersection()
{
    return static_cast<bool>((llflags >> 3) & 0x01);
}

 __device__ bool LLPolyVertex::GetIsValidIntersection()
{
    return static_cast<bool>((llflags >> 2) & 0x01);
}

 __device__ bool LLPolyVertex::GetIsEntry()
{
    return static_cast<bool>((llflags >> 1) & 0x01);
}

 __device__ bool LLPolyVertex::GetWasProcessed()
{
    return static_cast<bool>((llflags >> 0) & 0x01);
}

// Bit setters
__device__ bool LLPolyVertex::SetHasIntersections(bool flag)
{
    llflags = (llflags & 0xEF) | ((static_cast<uint8_t>(flag)) << 4);
}
__device__ void LLPolyVertex::SetIsIntersection(bool flag)
{
    llflags = (llflags & 0xF7) | ((static_cast<uint8_t>(flag)) << 3);
}

__device__ void LLPolyVertex::SetIsValidIntersection(bool flag)
{
    llflags = (llflags & 0xFB) | ((static_cast<uint8_t>(flag)) << 2);
}

__device__ void LLPolyVertex::SetIsEntry(bool flag)
{
    llflags = (llflags & 0xFD) | ((static_cast<uint8_t>(flag)) << 1);
}

__device__ void LLPolyVertex::SetWasProcessed(bool flag)
{
    llflags = (llflags & 0xFE) | ((static_cast<uint8_t>(flag)) << 0);
}

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

        retFail.SetIsIntersection(true);
        retFail.SetIsValidIntersection(false);
        retFail.SetIsEntry(false);
        retFail.SetWasProcessed(false);

        return retFail;
    }

    float dx = sA.latitude - sB.latitude;
    float dy = sA.longitude - sB.longitude;

    float alongA = (bdx * dy - bdy * dx) / axb;
    float alongB = (adx * dy - ady * dx) / axb;

    bool intersectionValidity = (alongA > 0 && alongA < 1 && alongB > 0 && alongB < 1);

    LLPolyVertex ret = {
        {sA.latitude + alongA * adx, sA.longitude + alongA * ady}, 0x0, alongA, alongB, -1, -1, -1};

    ret.SetHasIntersections(intersectionValidity);
    ret.SetIsIntersection(true);
    ret.SetIsValidIntersection(intersectionValidity);
    ret.SetIsEntry(false);
    ret.SetWasProcessed(false);

    return ret;
}

__global__ void kernel_calc_ll_buffers_size(int32_t* llPolygonABufferSizes,
                                            int32_t* llPolygonBBufferSizes,
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

        const int32_t polyIdxA = polygonA.PolyIdxAt(iAIdx);
        const int32_t polyCountA = polygonA.PolyCountAt(iAIdx);

        const int32_t polyIdxB = polygonB.PolyIdxAt(iBIdx);
        const int32_t polyCountB = polygonB.PolyCountAt(iBIdx);

        int32_t intersectCount = 0;
        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            const int32_t pointIdxA = polygonA.PointIdxAt(a);
            const int32_t pointCountA = polygonA.PointCountAt(a);

            int8_t intersectionPresentInSubPolygonA = 0;

            for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                const int32_t pointIdxB = polygonB.PointIdxAt(b);
                const int32_t pointCountB = polygonB.PointCountAt(b);

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

                        if (intersection.GetIsValidIntersection())
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
        const int32_t n = polygonA.TotalPointCountAt(iAIdx);
        const int32_t k = polygonB.TotalPointCountAt(iBIdx);

        // Assign the calculated buffers size
        llPolygonABufferSizes[i] = n + intersectCount;
        llPolygonBBufferSizes[i] = k + intersectCount;
    }
}

__global__ void kernel_build_ll(LLPolyVertex* llPolygonBuffers,
                                GPUMemory::GPUPolygon polygon,
                                int32_t* llPolygonBufferSizesPrefixSum,
                                int8_t* PolygonIntersectionPresenceFlags,
                                bool isConst,
                                int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t iIdx = isConst ? 0 : i;

        int32_t llPolygonEndIdx = 0;

        const int32_t polyIdx = polygon.PolyIdxAt(iIdx);
        const int32_t polyCount = polygon.PolyCountAt(iIdx);

        // Transform polygon
        for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
        {
            const int32_t pointIdx = polygon.PointIdxAt(p);
            const int32_t pointCount = polygon.PointCountAt(p);

            for (int32_t point = pointIdx; point < (pointIdx + pointCount); point++)
            {
                const int32_t localIdx = pointIdx - polygon.PointIdxAt(polyIdx);

                // Set the linked list entry
                LLPolyVertex vertex = {polygon.polyPoints[point],
                                       0x0,
                                       -1.0,
                                       -1.0,
                                       ((i == 0) ? 0 : llPolygonBufferSizesPrefixSum[i - 1]) +
                                           localIdx + (point - pointIdx - 1 + pointCount) % pointCount,
                                       ((i == 0) ? 0 : llPolygonBufferSizesPrefixSum[i - 1]) +
                                           localIdx + (point - pointIdx + 1) % pointCount,
                                       -1};

                vertex.SetHasIntersections(PolygonIntersectionPresenceFlags[p]);
                vertex.SetIsIntersection(false);
                vertex.SetIsValidIntersection(false);
                vertex.SetIsEntry(false);
                vertex.SetWasProcessed(false);

                llPolygonBuffers[((i == 0) ? 0 : llPolygonBufferSizesPrefixSum[i - 1]) + llPolygonEndIdx] = vertex;

                // Increment the local pointer to the end of the ll
                llPolygonEndIdx++;
            }
        }
    }
}

__global__ void kernel_add_and_crosslink_intersections_to_ll(LLPolyVertex* llPolygonABuffers,
                                                             LLPolyVertex* llPolygonBBuffers,
                                                             GPUMemory::GPUPolygon polygonA,
                                                             GPUMemory::GPUPolygon polygonB,
                                                             int32_t* llPolygonABufferSizesPrefixSum,
                                                             int32_t* llPolygonBBufferSizesPrefixSum,
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
        int32_t llPolygonAEndIdx = polygonA.TotalPointCountAt(iAIdx);
        int32_t llPolygonBEndIdx = polygonB.TotalPointCountAt(iBIdx);

        const int32_t polyIdxA = polygonA.PolyIdxAt(iAIdx);
        const int32_t polyCountA = polygonA.PolyCountAt(iAIdx);

        const int32_t polyIdxB = polygonB.PolyIdxAt(iBIdx);
        const int32_t polyCountB = polygonB.PolyCountAt(iBIdx);

        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            const int32_t pointIdxA = polygonA.PointIdxAt(a);
            const int32_t pointCountA = polygonA.PointCountAt(a);

            for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                const int32_t pointIdxB = polygonB.PointIdxAt(b);
                const int32_t pointCountB = polygonB.PointCountAt(b);

                // Calculate intersections and insert them into the ll
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
                        if (intersection.GetIsValidIntersection())
                        {
                            int32_t llPolygonAEndIdxLocal =
                                ((i == 0) ? 0 : llPolygonABufferSizesPrefixSum[i - 1]) + llPolygonAEndIdx;
                            int32_t llPolygonBEndIdxLocal =
                                ((i == 0) ? 0 : llPolygonBBufferSizesPrefixSum[i - 1]) + llPolygonBEndIdx;

                            // Save the intersection data
                            llPolygonABuffers[llPolygonAEndIdxLocal] = intersection;
                            llPolygonBBuffers[llPolygonBEndIdxLocal] = intersection;

                            // Write the cross reference indices
                            llPolygonABuffers[llPolygonAEndIdxLocal].crossIdx = llPolygonBEndIdxLocal;
                            llPolygonBBuffers[llPolygonBEndIdxLocal].crossIdx = llPolygonAEndIdxLocal;

                            // "Rewire" the prev and next pointers in both linked lists
                            // so that the point is in it's correct place
                            // according to the parametric distance from the beginning of the line segment
                            //////////////////////////////////////////////////////////////////////////////
                            // First polygon - A
                            const int32_t localIdxA = pointIdxA - polygonA.PointIdxAt(polyIdxA);

                            const int32_t begIdxA = ((i == 0) ? 0 : llPolygonABufferSizesPrefixSum[i - 1]) +
                                                    localIdxA + (pointA - pointIdxA) % pointCountA;
                            const int32_t endIdxA = ((i == 0) ? 0 : llPolygonABufferSizesPrefixSum[i - 1]) +
                                                    localIdxA + (pointA - pointIdxA + 1) % pointCountA;

                            int32_t nextIdxA = llPolygonABuffers[begIdxA].nextIdx;
                            while (nextIdxA != endIdxA && llPolygonABuffers[llPolygonAEndIdxLocal].distanceAlongA >
                                                              llPolygonABuffers[nextIdxA].distanceAlongA)
                            {
                                nextIdxA = llPolygonABuffers[nextIdxA].nextIdx;
                            }

                            // Rewire the pointers for the first polygon - A
                            llPolygonABuffers[llPolygonAEndIdxLocal].prevIdx =
                                llPolygonABuffers[nextIdxA].prevIdx;
                            llPolygonABuffers[llPolygonAEndIdxLocal].nextIdx = nextIdxA;

                            llPolygonABuffers[llPolygonABuffers[nextIdxA].prevIdx].nextIdx = llPolygonAEndIdxLocal;
                            llPolygonABuffers[nextIdxA].prevIdx = llPolygonAEndIdxLocal;
                            //////////////////////////////////////////////////////////////////////////////
                            // Second polygon - B
                            const int32_t localIdxB = pointIdxB - polygonB.PointIdxAt(polyIdxB);

                            const int32_t begIdxB = ((i == 0) ? 0 : llPolygonBBufferSizesPrefixSum[i - 1]) +
                                                    localIdxB + (pointB - pointIdxB) % pointCountB;
                            const int32_t endIdxB = ((i == 0) ? 0 : llPolygonBBufferSizesPrefixSum[i - 1]) +
                                                    localIdxB + (pointB - pointIdxB + 1) % pointCountB;

                            int32_t nextIdxB = llPolygonBBuffers[begIdxB].nextIdx;
                            while (nextIdxB != endIdxB && llPolygonBBuffers[llPolygonBEndIdxLocal].distanceAlongB >
                                                              llPolygonBBuffers[nextIdxB].distanceAlongB)
                            {
                                nextIdxB = llPolygonBBuffers[nextIdxB].nextIdx;
                            }

                            // Rewire the pointers for the second polygon - B
                            llPolygonBBuffers[llPolygonBEndIdxLocal].prevIdx =
                                llPolygonBBuffers[nextIdxB].prevIdx;
                            llPolygonBBuffers[llPolygonBEndIdxLocal].nextIdx = nextIdxB;

                            llPolygonBBuffers[llPolygonBBuffers[nextIdxB].prevIdx].nextIdx = llPolygonBEndIdxLocal;
                            llPolygonBBuffers[nextIdxB].prevIdx = llPolygonBEndIdxLocal;
                            //////////////////////////////////////////////////////////////////////////////

                            // Increment the ll end pointers
                            llPolygonAEndIdx++;
                            llPolygonBEndIdx++;
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

    const int32_t polyIdx = polygon.PolyIdxAt(idx);
    const int32_t polyCount = polygon.PolyCountAt(idx);

    for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
    {
        const int32_t pointIdx = polygon.PointIdxAt(p);
        const int32_t pointCount = polygon.PointCountAt(p);

        // Dank raycasting magic as seen in GPUPolygonContains
        for (int32_t point = pointIdx; point < (pointIdx + pointCount); point++)
        {
            const int32_t pBeg = point;
            const int32_t pEnd = pointIdx + (point - pointIdx + 1) % pointCount;

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

__global__ void kernel_label_intersections(LLPolyVertex* llPolygonBuffers,
                                           GPUMemory::GPUPolygon polygonPrimary,
                                           GPUMemory::GPUPolygon polygonSecondary,
                                           int32_t* llPolygonBufferSizesPrefixSum,
                                           bool isPrimaryConst,
                                           bool isSecondaryConst,
                                           int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int32_t iPrimary = isPrimaryConst ? 0 : i;
        const int32_t iSecondary = isSecondaryConst ? 0 : i;

        const int32_t polyIdx = polygonPrimary.PolyIdxAt(iPrimary);
        const int32_t polyCount = polygonPrimary.PolyCountAt(iPrimary);

        for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
        {
            const int32_t pointIdx = polygonPrimary.PointIdxAt(p);
            const int32_t pointCount = polygonPrimary.PointCountAt(p);

            // Iterate trough the linked list for the current sub polygon and label the intersections
            const int32_t localIdx = pointIdx - polygonPrimary.PointIdxAt(polyIdx);

            const int32_t begIdx = ((i == 0) ? 0 : llPolygonBufferSizesPrefixSum[i - 1]) + localIdx;
            const int32_t endIdx =
                ((i == 0) ? 0 : llPolygonBufferSizesPrefixSum[i - 1]) + localIdx + pointCount - 1;

            // Check the inclusion of the first point in the other polygon
            bool isPointInPolygon = !is_point_in_complex_polygon_at(llPolygonBuffers[begIdx].vertex,
                                                                    polygonSecondary, iSecondary);

            int32_t nextIdx = begIdx;
            do
            {
                // If the given vertex is an intersection - assign the correct entry/exit label
                if (llPolygonBuffers[nextIdx].GetIsIntersection())
                {
                    llPolygonBuffers[nextIdx].SetIsEntry(isPointInPolygon);
                    isPointInPolygon = !isPointInPolygon;
                }

                nextIdx = llPolygonBuffers[nextIdx].nextIdx;
            } while (nextIdx != begIdx);
        }
    }
}