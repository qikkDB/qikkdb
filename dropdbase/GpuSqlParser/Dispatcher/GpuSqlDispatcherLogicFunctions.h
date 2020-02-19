#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../../QueryEngine/GPUCore/GPUBinary.cuh"
#include "../../QueryEngine/GPUCore/GPUNullMask.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUFilterConditions.cuh"
#include "GpuSqlDispatcherVMFunctions.h"
#include <tuple>

/// Implementation of NOT operation dispatching
/// Implementation for column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LogicalNotCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "NotCol: " << colName << " " << reg << '\n';

    PointerAllocation column = allocatedPointers_.at(colName);
    int32_t retSize = column.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        int8_t* mask;
        if (column.GpuNullMaskPtr)
        {
            int8_t* nullMask;
            mask = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
        }
        else
        {
            mask = AllocateRegister<int8_t>(reg, retSize);
        }
        GPULogic::Not<int8_t, T*>(mask, reinterpret_cast<T*>(column.GpuPtr),
                                  reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), retSize);
    }

    FreeColumnIfRegister<T>(colName);
    return InstructionStatus::CONTINUE;
}

template <typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LogicalNotConst()
{
    return InstructionStatus::CONTINUE;
}


template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::NullMaskCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "NullMaskCol: " << colName << " " << reg << '\n';

    if (colName.front() == '$')
    {
        throw NullMaskOperationInvalidOperandException();
    }

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadColNullMask(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    PointerAllocation columnMask = allocatedPointers_.at(colName + NULL_SUFFIX);
    size_t nullMaskSize = (columnMask.ElementCount + 8 * sizeof(int8_t) - 1) / (8 * sizeof(int8_t));

    if (!IsRegisterAllocated(reg))
    {
        int8_t* outFilterMask;

        int8_t* nullMask;
        outFilterMask = AllocateRegister<int8_t>(reg, columnMask.ElementCount, &nullMask);
        GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(columnMask.GpuPtr), nullMaskSize);
        GPUNullMask::Col<OP>(outFilterMask, reinterpret_cast<int8_t*>(columnMask.GpuPtr),
                             nullMaskSize, columnMask.ElementCount);
    }
    return InstructionStatus::CONTINUE;
}
