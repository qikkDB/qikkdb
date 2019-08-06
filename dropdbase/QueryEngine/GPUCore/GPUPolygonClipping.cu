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
        LLPolyVertex retFail = {{0, 0}, true, false, -1.0, -1.0, -1, -1, -1};
        return retFail;
    }

    float dx = sA.latitude - sB.latitude;
    float dy = sA.longitude - sB.longitude;

    float alongA = (bdx * dy - bdy * dx) / axb;
    float alongB = (adx * dy - ady * dx) / axb;

    bool intersectionValidity = (alongA > 0 && alongA < 1 && alongB > 0 && alongB < 1);

    LLPolyVertex ret = {{sA.latitude + alongA * adx, sA.longitude + alongA * ady},
                        true,
                        intersectionValidity,
                        alongA,
                        alongB,
                        -1,
                        -1,
                        -1};

    return ret;
}

__global__ void kernel_calc_LL_buffers_size(int32_t* LLPolygonABufferSizes,
                                            int32_t* LLPolygonBBufferSizes,
                                            GPUMemory::GPUPolygon polygonA,
                                            GPUMemory::GPUPolygon polygonB,
                                            int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t polyIdxA = GPUMemory::PolyIdxAt(polygonA, i);
        int32_t polyCountA = GPUMemory::PolyCountAt(polygonA, i);

        int32_t polyIdxB = GPUMemory::PolyIdxAt(polygonB, i);
        int32_t polyCountB = GPUMemory::PolyCountAt(polygonB, i);

        int32_t intersectCount = 0;
        for (int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            int32_t pointIdxA = GPUMemory::PointIdxAt(polygonA, a);
            int32_t pointCountA = GPUMemory::PointCountAt(polygonA, a);

            for (int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                int32_t pointIdxB = GPUMemory::PointIdxAt(polygonB, b);
                int32_t pointCountB = GPUMemory::PointCountAt(polygonB, b);

                // Calculate intersections count
                for (int32_t pointA = pointIdxA; pointA < (pointIdxA + pointCountA); pointA++)
                {
                    for (int32_t pointB = pointIdxB; pointB < (pointIdxB + pointCountB); pointB++)
                    {
                        LLPolyVertex intersection =
                            calc_intersect(polygonA.polyPoints[pointA],
                                           polygonA.polyPoints[pointIdxA + (pointA - pointIdxA + 1) % pointCountA],
                                           polygonB.polyPoints[pointB],
                                           polygonB.polyPoints[pointIdxB + (pointB - pointIdxB + 1) % pointCountB]);

                        if (intersection.isValidIntersection)
                        {
                            intersectCount++;
                        }
                    }
                }
            }
        }

        // Get the complex polygon vertex counts n and k
        int32_t n = GPUMemory::TotalPointCountAt(polygonA, i);
        int32_t k = GPUMemory::TotalPointCountAt(polygonB, i);

        // Assign the calculated buffers size
        LLPolygonABufferSizes[i] = n + intersectCount;
        LLPolygonBBufferSizes[i] = k + intersectCount;
    }
}

__global__ void kernel_build_LL(LLPolyVertex* LLPolygonBuffers,
                                GPUMemory::GPUPolygon polygon,
                                int32_t* LLPolygonBufferSizesPrefixSum,
                                int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t LLPolygonEndIdx = 0;

        int32_t polyIdx = GPUMemory::PolyIdxAt(polygon, i);
        int32_t polyCount = GPUMemory::PolyCountAt(polygon, i);

        // Transform polygon
        for (int32_t p = polyIdx; p < (polyIdx + polyCount); p++)
        {
            int32_t pointIdx = GPUMemory::PointIdxAt(polygon, p);
            int32_t pointCount = GPUMemory::PointCountAt(polygon, p);

            for (int32_t point = pointIdx; point < (pointIdx + pointCount); point++)
            {
                int32_t localIdx = pointIdx - GPUMemory::PointIdxAt(polygon, polyIdx);

                // Set the linked list entry
                LLPolygonBuffers[((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + LLPolygonEndIdx] = {
                    polygon.polyPoints[point],
                    false,
                    false,
                    -1.0,
                    -1.0,
                    ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx + (point - pointIdx - 1 + pointCount) % pointCount,
                    ((i == 0) ? 0 : LLPolygonBufferSizesPrefixSum[i - 1]) + localIdx + (point - pointIdx + 1) % pointCount,
                    -1};

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
                                                             int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // "Pointers" to the element after the last valid element of the linked lists
        // They begin after the last non intersection e.g. poly vertex address
        int32_t LLPolygonAEndIdx = GPUMemory::TotalPointCountAt(polygonA, i);
        int32_t LLPolygonBEndIdx = GPUMemory::TotalPointCountAt(polygonB, i);

        int32_t polyIdxA = GPUMemory::PolyIdxAt(polygonA, i);
        int32_t polyCountA = GPUMemory::PolyCountAt(polygonA, i);

        int32_t polyIdxB = GPUMemory::PolyIdxAt(polygonB, i);
        int32_t polyCountB = GPUMemory::PolyCountAt(polygonB, i);

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

                        // TODO TODO TODO
                        // If an intersection is valid, insert it into the linked lists and create a cross reference
                        if (intersection.isValidIntersection)
                        {
                            // Save the intersection data
                            LLPolygonABuffers[((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) + LLPolygonAEndIdx] = intersection;
                            LLPolygonBBuffers[((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) + LLPolygonBEndIdx] = intersection;

                            // Write the cross reference indices
                            LLPolygonABuffers[((i == 0) ? 0 : LLPolygonABufferSizesPrefixSum[i - 1]) + LLPolygonAEndIdx].crossIdx = LLPolygonBEndIdx;
                            LLPolygonBBuffers[((i == 0) ? 0 : LLPolygonBBufferSizesPrefixSum[i - 1]) + LLPolygonBEndIdx].crossIdx = LLPolygonAEndIdx;

                            // "Rewire" the prev and next pointers s othat the point is in it's correct place
                            // according to the parametric distance from the beginning of the line segment

                            // TODO

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