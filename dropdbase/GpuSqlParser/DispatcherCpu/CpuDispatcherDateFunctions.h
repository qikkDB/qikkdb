#pragma once

#include "../CpuSqlDispatcher.h"

template <typename OP>
int32_t CpuSqlDispatcher::DateExtractCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<int64_t>(colName))
    {
        return 1;
    }

    // TODO ResultType
    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int32_t* resultMin = AllocateRegister<int32_t>(reg + "_min", 1, std::get<2>(colValMin));
    int32_t* resultMax = AllocateRegister<int32_t>(reg + "_max", 1, std::get<2>(colValMax));

    resultMin[0] = OP{}.operator()(reinterpret_cast<int64_t*>(std::get<0>(colValMin))[0]);
    resultMax[0] = OP{}.operator()(reinterpret_cast<int64_t*>(std::get<0>(colValMax))[0]);

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Where evaluation dateCol_min: " << colName << ", " << reg + "_min"
        << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Where evaluation dateCol_max: " << colName << ", " << reg + "_max"
        << ": " << resultMax[0] << '\n';

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::DateExtractConst()
{
    auto cnst = arguments_.Read<int64_t>();
    auto reg = arguments_.Read<std::string>();

    int32_t* resultMin = AllocateRegister<int32_t>(reg + "_min", 1, false);
    int32_t* resultMax = AllocateRegister<int32_t>(reg + "_max", 1, false);

    resultMin[0] = OP{}.operator()(cnst);
    resultMax[0] = OP{}.operator()(cnst);

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Where evaluation dateConst_min: " << reg + "_min"
                                                  << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Where evaluation dateConst_max: " << reg + "_max"
                                                  << ": " << resultMax[0] << '\n';

    return 0;
}