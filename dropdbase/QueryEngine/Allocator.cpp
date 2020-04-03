#include "Allocator.h"
#include "DeviceChanger.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

Allocator::Allocator(int32_t deviceId) : device_id_(deviceId)
{
}

int8_t* Allocator::Allocate(const size_t numBytes)
{
    DeviceChanger deviceChanger(device_id_);

    int8_t* pointer;
    cudaMalloc(&pointer, numBytes);
    allocated_pointers_.insert(pointer);
    return pointer;
}

void Allocator::Deallocate(int8_t* pointer)
{
    DeviceChanger deviceChanger(device_id_);

    cudaFree(pointer);
    allocated_pointers_.erase(pointer);
}

void Allocator::Clear()
{
    DeviceChanger deviceChanger(device_id_);

    for (int8_t* pointer : allocated_pointers_)
    {
        cudaFree(pointer);
    }
    allocated_pointers_.clear();
}
