#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../../QueryEngine/GPUCore/GPUNullMask.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUFilterConditions.cuh"
#include "GpuSqlDispatcherVMFunctions.h"
#include <tuple>

template <typename OP, typename L, typename R>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Filter()
{
    InstructionArgument<R> right = DispatcherInstructionHelper<R>::LoadInstructionArgument(*this);
    InstructionArgument<L> left = DispatcherInstructionHelper<L>::LoadInstructionArgument(*this);

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    if (std::get<2>(right) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(right);
    }

    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Filter: " << reg << '\n';

    if constexpr (std::is_pointer<L>::value && std::is_pointer<R>::value)
    {
        if (std::get<0>(left) && std::get<0>(right))
        {
            const int32_t retSize = std::min(std::get<1>(left).ElementCount, std::get<1>(right).ElementCount);
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr || std::get<1>(right).GpuNullMaskPtr;
            InstructionResult<int8_t> result = DispatcherInstructionHelper<int8_t>::AllocateInstructionResult(
                *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (std::get<1>(left).GpuNullMaskPtr && std::get<1>(right).GpuNullMaskPtr)
                    {
                        GPUArithmetic<ArithmeticOperations::bitwiseOr, int8_t, int8_t*, int8_t*>::Arithmetic(
                            std::get<1>(result), reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                            reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr), bitMaskSize);
                    }
                    else if (std::get<1>(left).GpuNullMaskPtr)
                    {
                        GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                      reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                    else
                    {
                        GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                      reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                }
                GPUFilter<OP, L, R>::Filter(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
                DispatcherInstructionHelper<int8_t>::StoreInstructionResult(result, *this, reg,
                                                                            retSize, allocateNullMask,
                                                                            {std::get<3>(left),
                                                                             std::get<3>(right)});
            }
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

    else if constexpr (std::is_pointer<L>::value)
    {
        if (std::get<0>(left))
        {
            const int32_t retSize = std::get<1>(left).ElementCount;
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr;
            InstructionResult<int8_t> result = DispatcherInstructionHelper<int8_t>::AllocateInstructionResult(
                *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUFilter<OP, L, R>::Filter(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
                DispatcherInstructionHelper<int8_t>::StoreInstructionResult(result, *this, reg,
                                                                            retSize, allocateNullMask,
                                                                            {std::get<3>(left),
                                                                             std::get<3>(right)});
            }
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
    }

    else if constexpr (std::is_pointer<R>::value)
    {
        if (std::get<0>(right))
        {
            const int32_t retSize = std::get<1>(right).ElementCount;
            const bool allocateNullMask = std::get<1>(right).GpuNullMaskPtr;
            InstructionResult<int8_t> result = DispatcherInstructionHelper<int8_t>::AllocateInstructionResult(
                *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUFilter<OP, L, R>::Filter(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
                DispatcherInstructionHelper<int8_t>::StoreInstructionResult(result, *this, reg,
                                                                            retSize, allocateNullMask,
                                                                            {std::get<3>(left),
                                                                             std::get<3>(right)});
            }
        }
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

    else
    {
        const int32_t retSize = GetBlockSize();
        if (retSize == 0)
        {
            return InstructionStatus::OUT_OF_BLOCKS;
        }

        InstructionResult<int8_t> result =
            DispatcherInstructionHelper<int8_t>::AllocateInstructionResult(*this, reg, retSize, false, {});
        if (std::get<0>(result))
        {
            GPUFilter<OP, L, R>::Filter(std::get<0>(result), std::get<0>(left), std::get<0>(right),
                                        nullptr, retSize);
            DispatcherInstructionHelper<int8_t>::StoreInstructionResult(result, *this, reg, retSize, false, {});
        }
    }

    return InstructionStatus::CONTINUE;
}

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
