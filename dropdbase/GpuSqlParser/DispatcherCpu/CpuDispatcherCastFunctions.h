#pragma once

#include "../CpuSqlDispatcher.h"

template <typename OUT, typename IN>
int32_t CpuSqlDispatcher::CastNumericCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    LoadCol<IN>(colName);

    // TODO ResultType
    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    OUT* resultMin = AllocateRegister<OUT>(reg + "_min", 1, std::get<2>(colValMin));
    OUT* resultMax = AllocateRegister<OUT>(reg + "_max", 1, std::get<2>(colValMax));

    resultMin[0] = static_cast<OUT>(reinterpret_cast<IN*>(std::get<0>(colValMin))[0]);
    resultMax[0] = static_cast<OUT>(reinterpret_cast<IN*>(std::get<0>(colValMax))[0]);

    std::cout << "Where evaluation castNumericCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation castNumericCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OUT, typename IN>
int32_t CpuSqlDispatcher::CastNumericConst()
{
    IN cnst = arguments_.Read<IN>();
    auto reg = arguments_.Read<std::string>();

    OUT* resultMin = AllocateRegister<OUT>(reg + "_min", 1, false);
    OUT* resultMax = AllocateRegister<OUT>(reg + "_max", 1, false);

    resultMin[0] = static_cast<OUT>(cnst);
    resultMax[0] = static_cast<OUT>(cnst);

    std::cout << "Where evaluation castNumericConst_min: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation castNumericConst_max: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}