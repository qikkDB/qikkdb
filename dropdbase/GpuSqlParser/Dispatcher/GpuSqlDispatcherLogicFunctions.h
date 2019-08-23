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

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::FilterColConst()
{
    U cnst = arguments_.Read<U>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Filter: " << colName << " " << reg << '\n';

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

        GPUFilter::Filter<OP, T*, U>(mask, reinterpret_cast<T*>(column.GpuPtr), cnst,
                                     reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), retSize);
    }

    FreeColumnIfRegister<T>(colName);
    return 0;
}

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::FilterConstCol()
{
    auto colName = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Filter: " << colName << " " << reg << '\n';

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

        GPUFilter::Filter<OP, T, U*>(mask, cnst, reinterpret_cast<U*>(column.GpuPtr),
                                     reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), retSize);
    }

    FreeColumnIfRegister<U>(colName);
    return 0;
}

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::FilterColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    PointerAllocation columnRight = allocatedPointers_.at(colNameRight);
    PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);
    int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

    if (!IsRegisterAllocated(reg))
    {
        if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
        {
            int8_t* combinedMask;
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            if (columnLeft.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            GPUFilter::Filter<OP, T*, U*>(mask, reinterpret_cast<T*>(columnLeft.GpuPtr),
                                          reinterpret_cast<U*>(columnRight.GpuPtr), combinedMask, retSize);
        }
        else
        {
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize);
            GPUFilter::Filter<OP, T*, U*>(mask, reinterpret_cast<T*>(columnLeft.GpuPtr),
                                          reinterpret_cast<U*>(columnRight.GpuPtr), nullptr, retSize);
        }
    }

    FreeColumnIfRegister<U>(colNameRight);
    FreeColumnIfRegister<T>(colNameLeft);
    return 0;
}

/// Implementation of genric filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::FilterConstConst()
{
    U constRight = arguments_.Read<U>();
    T constLeft = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    if (!IsRegisterAllocated(reg))
    {
        int8_t* mask = AllocateRegister<int8_t>(reg, database_->GetBlockSize());
        GPUFilter::Filter<OP, T, U>(mask, constLeft, constRight, nullptr, database_->GetBlockSize());
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::FilterStringColConst()
{
    std::string cnst = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
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
        GPUFilter::colConst<OP>(mask, std::get<0>(column), constString, nullBitMask, retSize);
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::FilterStringConstCol()
{
    auto colName = arguments_.Read<std::string>();
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
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
        GPUFilter::constCol<OP>(mask, constString, std::get<0>(column), nullBitMask, retSize);
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::FilterStringColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<std::string>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<std::string>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
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
            GPUFilter::colCol<OP>(mask, std::get<0>(columnLeft), std::get<0>(columnRight), combinedMask, retSize);
        }

        else
        {
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize);
            GPUFilter::colCol<OP>(mask, std::get<0>(columnLeft), std::get<0>(columnRight), nullptr, retSize);
        }
    }
    return 0;
}


template <typename OP>
int32_t GpuSqlDispatcher::FilterStringConstConst()
{
    std::string cnstRight = arguments_.Read<std::string>();
    std::string cnstLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "FilterStringConstConst: " << cnstLeft << " " << cnstRight << " " << reg << '\n';

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString constStringLeft = InsertConstStringGpu(cnstLeft);
        GPUMemory::GPUString constStringRight = InsertConstStringGpu(cnstRight);

        int8_t* mask = AllocateRegister<int8_t>(reg, database_->GetBlockSize());
        GPUFilter::constConst<OP>(mask, constStringLeft, constStringRight, database_->GetBlockSize());
    }
    return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::LogicalColConst()
{
    U cnst = arguments_.Read<U>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

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

        GPULogic::Logic<OP, T*, U>(mask, reinterpret_cast<T*>(column.GpuPtr), cnst,
                                     reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), retSize);
    }

    FreeColumnIfRegister<T>(colName);
    return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::LogicalConstCol()
{
    auto colName = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

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

        GPULogic::Logic<OP, T, U*>(mask, cnst, reinterpret_cast<U*>(column.GpuPtr),
                                     reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), retSize);
    }

    FreeColumnIfRegister<U>(colName);
    return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::LogicalColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Logical: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    PointerAllocation columnRight = allocatedPointers_.at(colNameRight);
    PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);

    int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

    if (!IsRegisterAllocated(reg))
    {
        if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
        {
            int8_t* combinedMask;
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            if (columnLeft.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            GPULogic::Logic<OP, T*, U*>(mask, reinterpret_cast<T*>(columnLeft.GpuPtr),
                                       reinterpret_cast<U*>(columnRight.GpuPtr), combinedMask, retSize);
        }
        else
        {
            int8_t* mask = AllocateRegister<int8_t>(reg, retSize);
            GPULogic::Logic<OP, T*, U*>(mask, reinterpret_cast<T*>(columnLeft.GpuPtr),
                                       reinterpret_cast<U*>(columnRight.GpuPtr), nullptr, retSize);
        }
    }

    FreeColumnIfRegister<U>(colNameRight);
    FreeColumnIfRegister<T>(colNameLeft);
    return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::LogicalConstConst()
{
    U constRight = arguments_.Read<U>();
    T constLeft = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    if (!IsRegisterAllocated(reg))
    {
        int8_t* mask = AllocateRegister<int8_t>(reg, database_->GetBlockSize());
        GPULogic::Logic<OP, T, U>(mask, constLeft, constRight, nullptr, database_->GetBlockSize());
    }

    return 0;
}

/// Implementation of NOT operation dispatching
/// Implementation for column case
/// Pops data from argument memory stream, and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T>
int32_t GpuSqlDispatcher::LogicalNotCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "NotCol: " << colName << " " << reg << '\n';

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
    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::LogicalNotConst()
{
    return 0;
}


template <typename OP>
int32_t GpuSqlDispatcher::NullMaskCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info) << "NullMaskCol: " << colName << " " << reg << '\n';

    if (colName.front() == '$')
    {
        throw NullMaskOperationInvalidOperandException();
    }

    int32_t loadFlag = LoadColNullMask(colName);
    if (loadFlag)
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
    return 0;
}
