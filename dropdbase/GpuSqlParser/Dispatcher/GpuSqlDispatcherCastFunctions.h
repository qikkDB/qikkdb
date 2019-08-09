#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUCast.cuh"

template <typename OUT, typename IN>
int32_t GpuSqlDispatcher::CastNumericCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<IN>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastNumericCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
        groupByColumns_.end())
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            OUT* result;

            if (column.GpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize);
            }

            GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(column.GpuPtr), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<OUT>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            OUT* result;
            if (column.GpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg, retSize);
            }
            GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(column.GpuPtr), retSize);
        }
    }

    FreeColumnIfRegister<IN>(colName);
    return 0;
}

template <typename OUT, typename IN>
int32_t GpuSqlDispatcher::CastNumericConst()
{
    IN cnst = arguments_.Read<IN>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastNumericConst: " << reg << '\n';

    int32_t retSize = 1;

    if (!IsRegisterAllocated(reg))
    {
        OUT* result = AllocateRegister<OUT>(reg, retSize);
        GPUCast::CastNumericConst(result, cnst, retSize);
    }
    return 0;
}