#include "GPUPolygonClipping.cuh"

__device__ LLPolyVertex calc_intersect(NativeGeoPoint sA, NativeGeoPoint eA,
                                       NativeGeoPoint sB, NativeGeoPoint eB)
{
    float adx = eA.latitude - sA.latitude;
    float ady = eA.longitude - sA.longitude;
    float bdx = eB.latitude - sB.latitude;
    float bdy = eB.longitude - sB.longitude;

    float axb = adx * bdy - ady * bdx;

    if (axb == 0)
    {
        LLPolyVertex retFail = {0, 0, true, false, -1, -1, -1, -1, -1};
        return retFail;
    }

    float dx = sA.latitude - sB.latitude;
    float dy = sA.longitude - sB.longitude;

    float alongA = (bdx * dy - bdy * dx) / axb;
    float alongB = (adx * dy - ady * dx) / axb;

    bool intersectionValidity = (alongA > 0 && alongA < 1 && alongB > 0 && alongB < 1);

    LLPolyVertex ret = {
        sA.latitude + alongA * adx,
        sA.longitude + alongA * ady,
        true,
        intersectionValidity,
        alongA,
        alongB,
        -1,
        -1,
        -1
    };

    return ret;
}

__global__ void kernel_calc_LL_buffers_size(int32_t *intesection_counts, 
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

        // The number of LL elements is the number of vertices of complex polygon A = n, 
        // complex polygon B = k and the number of intersections between them
        int32_t n = 0;
        int32_t k = 0;
        int32_t intersectCount = 0;
        for(int32_t a = polyIdxA; a < (polyIdxA + polyCountA); a++)
        {
            int32_t pointIdxA = GPUMemory::PointIdxAt(polygonA, a);
            int32_t pointCountA = GPUMemory::PointCountAt(polygonA, a);

            for(int32_t b = polyIdxB; b < (polyIdxB + polyCountB); b++)
            {
                int32_t pointIdxB = GPUMemory::PointIdxAt(polygonB, b);
                int32_t pointCountB = GPUMemory::PointCountAt(polygonB, b);

                // Calculate intersections count
                for(int32_t pointA = pointIdxA; pointA < (pointIdxA + pointCountA); pointA++)
                {
                    for(int32_t pointB = pointIdxB; pointB < (pointIdxB + pointCountB); pointB++)
                    {
                        LLPolyVertex result = calc_intersect(polygonA.polyPoints[pointA], 
                                                             polygonA.polyPoints[pointIdxA + (pointA + 1) % pointCountA],
                                                             polygonB.polyPoints[pointB],
                                                             polygonB.polyPoints[pointIdxB + (pointB + 1) % pointCountB]);
                        if(result.isValidIntersection)
                        {
                            intersectCount++;
                        }
                    }       
                }
            }   
        }

        // Get the complex polygon vertex counts n and k
        if(i == 0)
        {
            n = GPUMemory::PointIdxAt(polygonA, polyIdxA + polyCountA);
            k = GPUMemory::PointIdxAt(polygonB, polyIdxB + polyCountB);
        }
        else
        {
            int32_t polyIdxAPrev = GPUMemory::PolyIdxAt(polygonA, i - 1);
            int32_t polyCountAPrev = GPUMemory::PolyCountAt(polygonA, i - 1);
    
            int32_t polyIdxBPrev = GPUMemory::PolyIdxAt(polygonB, i - 1);
            int32_t polyCountBPrev = GPUMemory::PolyCountAt(polygonB, i - 1);

            n = GPUMemory::PointIdxAt(polygonA, polyIdxA + polyCountA) - GPUMemory::PointIdxAt(polygonA, polyIdxAPrev + polyCountAPrev);
            k = GPUMemory::PointIdxAt(polygonB, polyIdxB + polyCountB) - GPUMemory::PointIdxAt(polygonB, polyIdxBPrev + polyCountBPrev);
        }

        // Assign the calculated buffers size
        intesection_counts[i] = n + k + intersectCount;
    }
}
