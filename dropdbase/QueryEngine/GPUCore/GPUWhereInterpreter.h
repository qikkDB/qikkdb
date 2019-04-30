#pragma once

#include <cstdint>
#include "GPUStack.cuh"

struct GPUOpCode;

typedef void (*GpuVMFunction)(GPUOpCode, int32_t, GPUStack<2048>&, void**);

struct GPUOpCode
{
    GpuVMFunction fun_ptr;
	char data[sizeof(void*)];
};

__global__ void kernel_fill_gpu_dispatch_table(GpuVMFunction* gpuDispatchPtr, size_t arraySize);
