#pragma once

#include <cstdint>

constexpr int32_t OPERATIONS_COUNT = 10;
struct GPUOpCode;

typedef int8_t (*DispatchFunction)(GPUOpCode, int32_t);

struct GPUOpCode
{
    DispatchFunction fun_ptr;
    void* dataLeft;
    void* dataRight;
    int32_t regIdx;
};

void FillGpuDispatchTable(DispatchFunction* gpuDispatchTable, int32_t gpuDispatchTableSize);
