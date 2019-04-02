#pragma once

#include <cstdint>
#include "GPUStack.cuh"

constexpr int32_t OPERATIONS_COUNT = 22;
struct GPUOpCode;

typedef void (*DispatchFunction)(GPUOpCode, int32_t, GPUStack<2048>&, void**);

struct GPUOpCode
{
    DispatchFunction fun_ptr;
	char data[sizeof(void*)];
};

__global__ void kernel_fill_gpu_dispatch_table(DispatchFunction* gpuDispatchPtr, size_t arraySize);
