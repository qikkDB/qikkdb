#include "GPUMemory.cuh"
#include "../../GpuSqlParser/GpuSqlDispatcher.h"
#include <vector>
#include <string>

bool GPUMemory::EvictWithLockList()
{
    return Context::getInstance().getCacheForCurrentDevice().evict();
}

void GPUMemory::clear()
{
    Context::getInstance().GetAllocatorForCurrentDevice().Clear();
    CheckCudaError(cudaGetLastError());
}

void GPUMemory::free(void* p_block)
{
    Context::getInstance().GetAllocatorForCurrentDevice().Deallocate(static_cast<int8_t*>(p_block));
    CheckCudaError(cudaGetLastError());
}

void GPUMemory::free(GPUPolygon polygonCol)
{
    if (polygonCol.polyPoints)
    {
        GPUMemory::free(polygonCol.polyPoints);
    }

    if (polygonCol.pointIdx)
    {
        GPUMemory::free(polygonCol.pointIdx);
    }

    if (polygonCol.polyIdx)
    {
        GPUMemory::free(polygonCol.polyIdx);
    }
}

void GPUMemory::free(GPUString stringCol)
{
    if (stringCol.allChars != nullptr)
    {
        GPUMemory::free(stringCol.allChars);
    }

    if (stringCol.stringIndices != nullptr)
    {
        GPUMemory::free(stringCol.stringIndices);
    }
}

size_t GPUMemory::CalculateNullMaskSize(size_t dataElementCount, bool for32bit)
{
    if (for32bit)
    {
        return ((dataElementCount + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8)) *
               (sizeof(int32_t) / sizeof(int8_t));
    }
    else
    {
        return (dataElementCount + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
    }
}

template <>
void GPUMemory::PrintGpuBuffer<NativeGeoPoint>(const char* title, NativeGeoPoint* bufferGpu, int32_t dataElementCount)
{
    std::unique_ptr<NativeGeoPoint[]> bufferCpu(new NativeGeoPoint[dataElementCount]);
    GPUMemory::copyDeviceToHost(bufferCpu.get(), bufferGpu, dataElementCount);

    std::cout << title << " (" << reinterpret_cast<uintptr_t>(bufferGpu) << "): ";

    for (int32_t i = 0; i < dataElementCount; i++)
    {
        std::cout << "{" << bufferCpu[i].latitude << ", " << bufferCpu[i].longitude << "}"
                  << " ";
    }
    std::cout << std::endl;
}
