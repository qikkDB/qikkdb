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
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DateExtractCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<int64_t>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ExtractDatePartCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            int32_t* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<int32_t>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<int32_t>(reg + KEYS_SUFFIX, retSize);
            }
            GPUDate::Extract<OP>(result, reinterpret_cast<int64_t*>(column.GpuPtr), retSize);
            groupByColumns_.push_back({reg, COLUMN_INT});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            int32_t* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<int32_t>(reg, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<int32_t>(reg, retSize);
            }
            GPUDate::Extract<OP>(result, reinterpret_cast<int64_t*>(column.GpuPtr), retSize);
        }
    }

    FreeColumnIfRegister<int64_t>(colName);
    return InstructionStatus::CONTINUE;
}

/// Implementation of generic date part extract function dispatching given by the functor OP
/// Implementation for constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DateExtractConst()
{
    int64_t cnst = arguments_.Read<int64_t>();
    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ExtractDatePartConst: " << cnst << " " << reg << '\n';

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (!IsRegisterAllocated(reg))
    {
        int32_t* result = AllocateRegister<int32_t>(reg, retSize);
        GPUDate::Extract<OP>(result, cnst, retSize);
    }
    return InstructionStatus::CONTINUE;
}
