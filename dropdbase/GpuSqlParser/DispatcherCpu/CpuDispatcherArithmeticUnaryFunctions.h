#pragma once

#include "../CpuSqlDispatcher.h"

template <typename OP, typename T>
int32_t CpuSqlDispatcher::ArithmeticUnaryCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    typedef typename std::conditional<OP::isFloatRetType, float, T>::type ResultType;

    if (LoadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    ResultType* resultMin =
        AllocateRegister<ResultType>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
    ResultType* resultMax =
        AllocateRegister<ResultType>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);

    resultMin[0] = OP{}.template operator()<ResultType, T>(reinterpret_cast<T*>(std::get<0>(colValMin))[0]);
    resultMax[0] = OP{}.template operator()<ResultType, T>(reinterpret_cast<T*>(std::get<0>(colValMax))[0]);

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Where evaluation arithmeticUnaryCol_min: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0]
        << ", " << reg + "_min"
        << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Where evaluation arithmeticUnaryCol_max: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0]
        << ", " << reg + "_max"
        << ": " << resultMax[0] << '\n';

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::ArithmeticUnaryConst()
{
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    typedef typename std::conditional<OP::isFloatRetType, float, T>::type ResultType;

    ResultType* resultMin = AllocateRegister<ResultType>(reg + "_min", 1, !OP::isMonotonous);
    ResultType* resultMax = AllocateRegister<ResultType>(reg + "_max", 1, !OP::isMonotonous);

    resultMin[0] = OP{}.template operator()<ResultType, T>(cnst);
    resultMax[0] = OP{}.template operator()<ResultType, T>(cnst);

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Where evaluation arithmeticUnaryConst_min: " << reg + "_min"
        << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Where evaluation arithmeticUnaryConst_max: " << reg + "_max"
        << ": " << resultMax[0] << '\n';

    return 0;
}