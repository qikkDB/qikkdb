#include "GPUMemory.cuh"

__device__ int32_t GPUMemory::GPUPolygon::PointIdxAt(int32_t idx)
{
    return (idx == 0) ? 0 : pointIdx[idx - 1];
}

__device__ int32_t GPUMemory::GPUPolygon::PolyIdxAt(int32_t idx)
{
    return (idx == 0) ? 0 : polyIdx[idx - 1];
}

__device__ int32_t GPUMemory::GPUPolygon::PointCountAt(int32_t idx)
{
    return static_cast<int32_t>(pointIdx[idx] - PointIdxAt(idx));
}

__device__ int32_t GPUMemory::GPUPolygon::PolyCountAt(int32_t idx)
{
    return static_cast<int32_t>(polyIdx[idx] - PolyIdxAt(idx));
}

__device__ __host__ int32_t GPUMemory::GPUPolygon::TotalPointCountAt(int32_t idx)
{
    int32_t polyIdx = PolyIdxAt(idx);
    int32_t polyCount = PolyCountAt(idx);

    if (idx == 0)
    {
        return PointIdxAt(polyIdx + polyCount);
    }
    else
    {
        int32_t polyIdxPrev = PolyIdxAt(idx - 1);
        int32_t polyCountPrev = PolyCountAt(idx - 1);

        return PointIdxAt(polyIdx + polyCount) -
               PointIdxAt(polyIdxPrev + polyCountPrev);
    }
}