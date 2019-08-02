#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUDate.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"

/// Implementation of generic date part extract function dispatching given by the functor OP
/// Implementation for column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP>
int32_t GpuSqlDispatcher::dateExtractCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<int64_t>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "ExtractDatePartCol: " << colName << " " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            PointerAllocation column = allocatedPointers.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.elementCount;
            int32_t* result;
            if (column.gpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = allocateRegister<int32_t>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = allocateRegister<int32_t>(reg + KEYS_SUFFIX, retSize);
            }
            GPUDate::extractCol<OP>(result, reinterpret_cast<int64_t*>(column.gpuPtr), retSize);
            groupByColumns.push_back({reg, COLUMN_INT});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        PointerAllocation column = allocatedPointers.at(colName);
        int32_t retSize = column.elementCount;

        if (!isRegisterAllocated(reg))
        {
            int32_t* result;
            if (column.gpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = allocateRegister<int32_t>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = allocateRegister<int32_t>(reg, retSize);
            }
            GPUDate::extractCol<OP>(result, reinterpret_cast<int64_t*>(column.gpuPtr), retSize);
        }
    }

    freeColumnIfRegister<int64_t>(colName);
    return 0;
}

/// Implementation of generic date part extract function dispatching given by the functor OP
/// Implementation for constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP>
int32_t GpuSqlDispatcher::dateExtractConst()
{
    int64_t cnst = arguments.read<int64_t>();
    auto reg = arguments.read<std::string>();
    std::cout << "ExtractDatePartConst: " << cnst << " " << reg << std::endl;

    int32_t retSize = 1;

    if (!isRegisterAllocated(reg))
    {
        int32_t* result = allocateRegister<int32_t>(reg, retSize);
        GPUDate::extractConst<OP>(result, cnst, retSize);
    }
    return 0;
}
