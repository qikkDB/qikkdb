#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "cuda_ptr.h"
#include "GPUReconstruct.cuh"
#include "GPUMemory.cuh"

__global__ void kernel_fill_date_string(GPUMemory::GPUString outCol,
                                        int32_t* years,
                                        int32_t* months,
                                        int32_t* days,
                                        int32_t* hours,
                                        int32_t* minutes,
                                        int32_t* seconds,
                                        int32_t dataElementCount);
