#include "GPUMemory.cuh"

__device__ __host__ int32_t GPUMemory::PointIdxAt(GPUMemory::GPUPolygon &polygon, int32_t idx) 
{ 
    return (idx == 0) ? 0 : polygon.pointIdx[idx - 1]; 
}

__device__ __host__ int32_t GPUMemory::PolyIdxAt(GPUMemory::GPUPolygon &polygon, int32_t idx) 
{ 
    return (idx == 0) ? 0 : polygon.polyIdx[idx - 1]; 
}

__device__ __host__ int32_t GPUMemory::PointCountAt(GPUMemory::GPUPolygon &polygon, int32_t idx) 
{ 
    return static_cast<int32_t>(polygon.pointIdx[idx] - PointIdxAt(polygon, idx)); 
}

__device__ __host__ int32_t GPUMemory::PolyCountAt(GPUMemory::GPUPolygon &polygon, int32_t idx) 
{ 
    return static_cast<int32_t>(polygon.polyIdx[idx] - PolyIdxAt(polygon, idx)); 
}
