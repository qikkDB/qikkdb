#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename L, typename R>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticColConst()
{
    std::tuple<R, PointerAllocation, InstructionStatus, std::string> right = LoadInstructionArgument<R>();
    std::tuple<L, PointerAllocation, InstructionStatus, std::string> left = LoadInstructionArgument<L>();

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    if (std::get<2>(right) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(right);
    }

    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Arithmetic: " << reg << '\n';

    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<std::remove_pointer<L>::type>::value &&
            std::is_floating_point<std::remove_pointer<R>::type>::value ||
        std::is_integral<std::remove_pointer<L>::type>::value &&
            std::is_integral<std::remove_pointer<R>::type>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral,
        typename std::conditional<sizeof(std::remove_pointer<L>::type) >= sizeof(std::remove_pointer<R>::type),
                                  std::remove_pointer<L>::type, std::remove_pointer<R>::type>::type,
        typename std::conditional<std::is_floating_point<std::remove_pointer<L>::type>::value, std::remove_pointer<L>::type,
                                  typename std::conditional<std::is_floating_point<std::remove_pointer<R>::type>::value,
                                                            std::remove_pointer<R>::type, void>::type>::type>::type ResultType;

    if (std::is_pointer<L>::value && std::is_pointer<R>::value)
    {
        if (std::get<0>(left) && std::get<0>(right))
        {
            const int32_t retSize = std::min(std::get<1>(left).ElementCount, std::get<1>(right).ElementCount);
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr || std::get<1>(right).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                std::get<1>(result), reinterpret_cast<int8_t*>(std::get<0>(left).GpuNullMaskPtr),
                reinterpret_cast<int8_t*>(std::get<0>(right).GpuNullMaskPtr), bitMaskSize);
            GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                            std::get<0>(right), retSize);
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

    else if (std::is_pointer<L>::value)
    {
        if (std::get<0>(left))
        {
            const int32_t retSize = std::get<1>(left).ElementCount;
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                          reinterpret_cast<int8_t*>(std::get<0>(left).GpuNullMaskPtr),
                                          bitMaskSize);
            GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                            std::get<0>(right), retSize);
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
    }

    else if (std::is_pointer<R>::value)
    {
        if (std::get<0>(right))
        {
            const int32_t retSize = std::get<1>(right).ElementCount;
            const bool allocateNullMask = std::get<1>(right).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                          reinterpret_cast<int8_t*>(std::get<0>(right).GpuNullMaskPtr),
                                          bitMaskSize);
            GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                            std::get<0>(right), retSize);
        }
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

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
                int8_t* nullMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
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
                int8_t* nullMask;
                result = AllocateRegister<ResultType>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
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
                int8_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnLeft.GpuNullMaskPtr)
            {
                int8_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                int8_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
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
                int8_t* combinedMask;
                result = AllocateRegister<ResultType>(reg + KEYS_SUFFIX, retSize, &combinedMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
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
                int8_t* combinedMask;
                result = AllocateRegister<ResultType>(reg, retSize, &combinedMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
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
