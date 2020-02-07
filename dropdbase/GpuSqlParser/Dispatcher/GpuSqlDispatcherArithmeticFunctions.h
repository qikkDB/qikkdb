#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticColConst()
{
    U cnst = arguments_.Read<U>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
        std::is_integral<T>::value && std::is_integral<U>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;
    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ArithmeticColConst: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            ResultType* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T*, U>(result, reinterpret_cast<T*>(column.GpuPtr),
                                                             cnst, retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<ResultType>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;
        if (!IsRegisterAllocated(reg))
        {
            ResultType* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<ResultType>(reg, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<ResultType>(reg, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T*, U>(result, reinterpret_cast<T*>(column.GpuPtr),
                                                             cnst, retSize);
        }
    }
    FreeColumnIfRegister<T>(colName);
    return InstructionStatus::CONTINUE;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticConstCol()
{
    auto colName = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();


    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
        std::is_integral<T>::value && std::is_integral<U>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;
    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ArithmeticConstCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            ResultType* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T, U*>(result, cnst,
                                                             reinterpret_cast<U*>(column.GpuPtr), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<ResultType>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            ResultType* result;
            if (column.GpuNullMaskPtr)
            {
                int64_t* nullMask;
                result = AllocateRegister<ResultType>(reg, retSize, &nullMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(column.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<ResultType>(reg, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T, U*>(result, cnst,
                                                             reinterpret_cast<U*>(column.GpuPtr), retSize);
        }
    }
    FreeColumnIfRegister<U>(colName);
    return InstructionStatus::CONTINUE;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();
    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
        std::is_integral<T>::value && std::is_integral<U>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colNameRight)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation columnRight = allocatedPointers_.at(colNameRight + KEYS_SUFFIX);
            PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);
            int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

            ResultType* result;
            if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
            {
                int64_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnLeft.GpuNullMaskPtr)
            {
                int64_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                int64_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr),
                                              bitMaskSize);
            }
            else
            {
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T*, U*>(result,
                                                              reinterpret_cast<T*>(columnLeft.GpuPtr),
                                                              reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<ResultType>()});
        }
    }
    else if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                          StringDataTypeComp(colNameLeft)) != groupByColumns_.end() &&
             !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation columnRight = allocatedPointers_.at(colNameRight);
            PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft + KEYS_SUFFIX);
            int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

            ResultType* result;
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int64_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T*, U*>(result,
                                                              reinterpret_cast<T*>(columnLeft.GpuPtr),
                                                              reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<ResultType>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation columnRight = allocatedPointers_.at(colNameRight);
        PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);
        int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

        if (!IsRegisterAllocated(reg))
        {
            ResultType* result;
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int64_t* combinedMask;
                result = AllocateRegister<ResultType>(reg, retSize, &combinedMask);
                int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                result = AllocateRegister<ResultType>(reg, retSize);
            }
            GPUArithmetic::Arithmetic<OP, ResultType, T*, U*>(result,
                                                              reinterpret_cast<T*>(columnLeft.GpuPtr),
                                                              reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
        }
    }
    FreeColumnIfRegister<T>(colNameLeft);
    FreeColumnIfRegister<U>(colNameRight);
    return InstructionStatus::CONTINUE;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticConstConst()
{
    U constRight = arguments_.Read<U>();
    T constLeft = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();
    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
        std::is_integral<T>::value && std::is_integral<U>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ArithmeticConstConst: " << reg << '\n';

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }

    if (!IsRegisterAllocated(reg))
    {
        ResultType* result = AllocateRegister<ResultType>(reg, retSize);
        GPUArithmetic::Arithmetic<OP, ResultType, T, U>(result, constLeft, constRight, retSize);
    }
    return InstructionStatus::CONTINUE;
}
