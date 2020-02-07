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
    InstructionArgument<R> right = InstructionArgumentLoadHelper<R>::LoadInstructionArgument(*this);
    InstructionArgument<L> left = InstructionArgumentLoadHelper<L>::LoadInstructionArgument(*this);

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
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (std::get<1>(left).GpuNullMaskPtr && std::get<1>(right).GpuNullMaskPtr)
                    {
                        GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
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
                GPUFilter::Filter<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
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
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUFilter::Filter<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
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
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUFilter::Filter<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                            std::get<0>(right), std::get<1>(result), retSize);
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

        std::pair<int8_t*, int8_t*> result = AllocateInstructionResult<int8_t>(reg, retSize, false, {});
        if (std::get<0>(result))
        {
            GPUFilter::Filter<OP, L, R>(std::get<0>(result), std::get<0>(left), std::get<0>(right),
                                        nullptr, retSize);
        }
    }

    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::FilterStringColConst()
{
    std::string cnst = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "FilterStringColConst: " << colName << " " << cnst << " " << reg << '\n';

    auto column = FindStringColumn(colName);
    int32_t retSize = std::get<1>(column);
    int8_t* nullBitMask = std::get<2>(column);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString constString = InsertConstStringGpu(cnst);
        int8_t* mask;
        if (nullBitMask)
        {
            int8_t* nullMask;
            mask = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, nullBitMask, bitMaskSize);
        }
        else
        {
            mask = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUFilter::FilterString<OP>(mask, std::get<0>(column), true, constString, false, nullBitMask, retSize);
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::FilterStringConstCol()
{
    auto colName = arguments_.Read<std::string>();
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "FilterStringConstCol: " << cnst << " " << colName << " " << reg << '\n';

    std::tuple<GPUMemory::GPUString, int32_t, int8_t*> column = FindStringColumn(colName);
    int32_t retSize = std::get<1>(column);
    int8_t* nullBitMask = std::get<2>(column);
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString constString = InsertConstStringGpu(cnst);
        int8_t* mask;
        if (nullBitMask)
        {
            int8_t* nullMask;
            mask = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, nullBitMask, bitMaskSize);
        }
        else
        {
            mask = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUFilter::FilterString<OP>(mask, constString, false, std::get<0>(column), true, nullBitMask, retSize);
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::FilterStringColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<std::string>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "FilterStringColCol: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    std::tuple<GPUMemory::GPUString, int32_t, int8_t*> columnLeft = FindStringColumn(colNameLeft);
    std::tuple<GPUMemory::GPUString, int32_t, int8_t*> columnRight = FindStringColumn(colNameRight);
    int32_t retSize = std::max(std::get<1>(columnLeft), std::get<1>(columnRight));
    int8_t* leftMask = std::get<2>(columnLeft);
    int8_t* rightMask = std::get<2>(columnRight);
    if (!IsRegisterAllocated(reg))
    {
        if (leftMask || rightMask)
        {
            int8_t* combinedMask;
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (leftMask && rightMask)
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(combinedMask, leftMask,
                                                                           rightMask, bitMaskSize);
            }
            if (leftMask)
            {
                GPUMemory::copyDeviceToDevice(combinedMask, leftMask, bitMaskSize);
            }
            else if (rightMask)
            {
                GPUMemory::copyDeviceToDevice(combinedMask, rightMask, bitMaskSize);
            }
            GPUFilter::FilterString<OP>(mask, std::get<0>(columnLeft), true,
                                        std::get<0>(columnRight), true, combinedMask, retSize);
        }

        else
        {
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize);
            GPUFilter::FilterString<OP>(mask, std::get<0>(columnLeft), true,
                                        std::get<0>(columnRight), true, nullptr, retSize);
        }
    }
    return InstructionStatus::CONTINUE;
}


template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::FilterStringConstConst()
{
    std::string cnstRight = arguments_.Read<std::string>();
    std::string cnstLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "FilterStringConstConst: " << cnstLeft << " " << cnstRight << " " << reg << '\n';
    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString constStringLeft = InsertConstStringGpu(cnstLeft);
        GPUMemory::GPUString constStringRight = InsertConstStringGpu(cnstRight);

        int8_t* mask = AllocateRegister<int8_t>(reg, retSize);
        GPUFilter::FilterString<OP>(mask, constStringLeft, false, constStringRight, false, nullptr, retSize);
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename L, typename R>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Logical()
{
    InstructionArgument<R> right = InstructionArgumentLoadHelper<R>::LoadInstructionArgument(*this);
    InstructionArgument<L> left = InstructionArgumentLoadHelper<L>::LoadInstructionArgument(*this);

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    if (std::get<2>(right) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(right);
    }

    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Logical: " << reg << '\n';

    if constexpr (std::is_pointer<L>::value && std::is_pointer<R>::value)
    {
        if (std::get<0>(left) && std::get<0>(right))
        {
            const int32_t retSize = std::min(std::get<1>(left).ElementCount, std::get<1>(right).ElementCount);
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr || std::get<1>(right).GpuNullMaskPtr;
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (std::get<1>(left).GpuNullMaskPtr && std::get<1>(right).GpuNullMaskPtr)
                    {
                        GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
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
                GPULogic::Logic<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                          std::get<0>(right), std::get<1>(result), retSize);
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
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPULogic::Logic<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                          std::get<0>(right), std::get<1>(result), retSize);
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
            std::pair<int8_t*, int8_t*> result =
                AllocateInstructionResult<int8_t>(reg, retSize, allocateNullMask,
                                                  {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPULogic::Logic<OP, L, R>(std::get<0>(result), std::get<0>(left),
                                          std::get<0>(right), std::get<1>(result), retSize);
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

        std::pair<int8_t*, int8_t*> result = AllocateInstructionResult<int8_t>(reg, retSize, false, {});
        if (std::get<0>(result))
        {
            GPULogic::Logic<OP, L, R>(std::get<0>(result), std::get<0>(left), std::get<0>(right), nullptr, retSize);
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
