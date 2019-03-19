#pragma once

#include <cstdint>
#include "GPUStack.cuh"

constexpr int32_t OPERATIONS_COUNT = 9;
struct GPUOpCode;

typedef int8_t (*DispatchFunction)(GPUOpCode, int32_t, GPUStack<2048>&, int8_t*, void**);

struct GPUOpCode
{
    DispatchFunction fun_ptr;
	char data[sizeof(void*)];
};