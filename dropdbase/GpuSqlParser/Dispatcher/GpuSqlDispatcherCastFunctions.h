#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUCast.cuh"

template <typename OUT, typename IN>
int32_t GpuSqlDispatcher::castNumericCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<IN>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "CastNumericCol: " << colName << " " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            PointerAllocation column = allocatedPointers.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.elementCount;
            OUT* result;

            if (column.gpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = allocateRegister<OUT>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = allocateRegister<OUT>(reg + KEYS_SUFFIX, retSize);
            }

            GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(column.gpuPtr), retSize);
            groupByColumns.push_back({reg, ::GetColumnType<OUT>()});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        PointerAllocation column = allocatedPointers.at(colName);
        int32_t retSize = column.elementCount;

        if (!isRegisterAllocated(reg))
        {
            OUT* result;
            if (column.gpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = allocateRegister<OUT>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = allocateRegister<OUT>(reg, retSize);
            }
            GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(column.gpuPtr), retSize);
        }
    }

    freeColumnIfRegister<IN>(colName);
    return 0;
}

template <typename OUT, typename IN>
int32_t GpuSqlDispatcher::castNumericConst()
{
    IN cnst = arguments.read<IN>();
    auto reg = arguments.read<std::string>();

    std::cout << "CastNumericConst: " << reg << std::endl;

    int32_t retSize = 1;

    if (!isRegisterAllocated(reg))
    {
        OUT* result = allocateRegister<OUT>(reg, retSize);
        GPUCast::CastNumericConst(result, cnst, retSize);
    }
    return 0;
}