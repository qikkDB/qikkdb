#pragma once

#include "../CpuSqlDispatcher.h"

template <typename OUT, typename IN>
int32_t CpuSqlDispatcher::castNumericCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    loadCol<IN>(colName);

    // TODO ResultType
    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    OUT* resultMin = allocateRegister<OUT>(reg + "_min", 1, std::get<2>(colValMin));
    OUT* resultMax = allocateRegister<OUT>(reg + "_max", 1, std::get<2>(colValMax));

    resultMin[0] = static_cast<OUT>(reinterpret_cast<IN*>(std::get<0>(colValMin))[0]);
    resultMax[0] = static_cast<OUT>(reinterpret_cast<IN*>(std::get<0>(colValMax))[0]);

    std::cout << "Where evaluation castNumericCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation castNumericCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OUT, typename IN>
int32_t CpuSqlDispatcher::castNumericConst()
{
    IN cnst = arguments.read<IN>();
    auto reg = arguments.read<std::string>();

    OUT* resultMin = allocateRegister<OUT>(reg + "_min", 1, false);
    OUT* resultMax = allocateRegister<OUT>(reg + "_max", 1, false);

    resultMin[0] = static_cast<OUT>(cnst);
    resultMax[0] = static_cast<OUT>(cnst);

    std::cout << "Where evaluation castNumericConst_min: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation castNumericConst_max: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}