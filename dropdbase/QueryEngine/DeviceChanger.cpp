#include "DeviceChanger.h"

#include <cuda.h>
#include <cuda_runtime.h>


DeviceChanger::DeviceChanger(int32_t newDevideId)
{
    cudaGetDevice(&old_device_id_);
    cudaSetDevice(newDevideId);
}

DeviceChanger::~DeviceChanger()
{
    cudaSetDevice(old_device_id_);
}
