#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T>
int32_t GpuSqlDispatcher::ArithmeticUnaryCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    // TODO STD conditional :: if OP == abs return type = T

    typedef typename std::conditional<OP::isFloatRetType, float, T>::type ResultType;

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "ArithmeticUnaryCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
        groupByColumns_.end())
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
            GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(column.GpuPtr), retSize);
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
            GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(column.GpuPtr), retSize);
        }
    }
    FreeColumnIfRegister<T>(colName);
    return 0;
}

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T>
int32_t GpuSqlDispatcher::ArithmeticUnaryConst()
{
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    // TODO STD conditional :: if OP == abs return type = T
    typedef typename std::conditional<OP::isFloatRetType, float, T>::type ResultType;

    CudaLogBoost::getInstance(CudaLogBoost::info) << "ArithmeticUnaryConst: " << reg << '\n';

    int32_t retSize = 1;

    if (!IsRegisterAllocated(reg))
    {
        ResultType* result = AllocateRegister<ResultType>(reg, retSize);
        GPUArithmeticUnary::cnst<OP, ResultType, T>(result, cnst, retSize);
    }

    return 0;
}