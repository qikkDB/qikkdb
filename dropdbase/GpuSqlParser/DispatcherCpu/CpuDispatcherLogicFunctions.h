#pragma once

#include "../CpuSqlDispatcher.h"
#include "CpuFilterInterval.h"
#include "QueryEngine/GPUCore/LogicOperations.h"
#include <tuple>

template <typename OP>
int32_t CpuSqlDispatcher::FilterStringColConst()
{
    auto colName = arguments_.Read<std::string>();
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<std::string>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

    switch (OP::interval)
    {
    case CpuFilterInterval::OUTER:
    case CpuFilterInterval::INNER:
        maskMin[0] = cnst >= reinterpret_cast<char*>(std::get<0>(colValMin)) &&
                     cnst <= reinterpret_cast<char*>(std::get<0>(colValMax));
        maskMax[0] = cnst >= reinterpret_cast<char*>(std::get<0>(colValMin)) &&
                     cnst <= reinterpret_cast<char*>(std::get<0>(colValMax));
        break;

    case CpuFilterInterval::NONE:
    default:
        maskMin[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                         std::get<1>(colValMin) - 1, cnst.c_str(), cnst.size());
        maskMax[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                         std::get<1>(colValMax) - 1, cnst.c_str(), cnst.size());
        break;
    }

    std::cout << "Where evaluation filterStringColConstMin: "
              << reinterpret_cast<char*>(std::get<0>(colValMin)) << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterStringColConstMax: "
              << reinterpret_cast<char*>(std::get<0>(colValMax)) << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::FilterStringConstCol()
{
    std::string cnst = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<std::string>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

    switch (OP::interval)
    {
    case CpuFilterInterval::OUTER:
    case CpuFilterInterval::INNER:
        maskMin[0] = cnst >= reinterpret_cast<char*>(std::get<0>(colValMin)) &&
                     cnst <= reinterpret_cast<char*>(std::get<0>(colValMax));
        maskMax[0] = cnst >= reinterpret_cast<char*>(std::get<0>(colValMin)) &&
                     cnst <= reinterpret_cast<char*>(std::get<0>(colValMax));
        break;

    case CpuFilterInterval::NONE:
    default:
        maskMin[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                         std::get<1>(colValMin) - 1, cnst.c_str(), cnst.size());
        maskMax[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                         std::get<1>(colValMax) - 1, cnst.c_str(), cnst.size());
        break;
    }

    std::cout << "Where evaluation filterStringConstColMin: "
              << reinterpret_cast<char*>(std::get<0>(colValMin)) << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterStringConstColMax: "
              << reinterpret_cast<char*>(std::get<0>(colValMax)) << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::FilterStringColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (LoadCol<std::string>(colNameLeft) || LoadCol<std::string>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = GetPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = GetPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers_.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers_.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers_.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers_.at(colPointerNameRightMax);

        maskMin = AllocateRegister<int8_t>(reg + "_min", 1,
                                           std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1,
                                           std::get<2>(colValLeftMax) || std::get<2>(colValRightMax));

        maskMin[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValLeftMin)),
                                         std::get<1>(colValLeftMin) - 1,
                                         reinterpret_cast<char*>(std::get<0>(colValRightMin)),
                                         std::get<1>(colValRightMin) - 1);
        maskMax[0] = OP{}.compareStrings(reinterpret_cast<char*>(std::get<0>(colValLeftMax)),
                                         std::get<1>(colValLeftMax) - 1,
                                         reinterpret_cast<char*>(std::get<0>(colValRightMax)),
                                         std::get<1>(colValRightMax) - 1);
    }

    std::cout << "Where evaluation filterStringColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterStringColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;
    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::FilterStringConstConst()
{
    std::string cnstLeft = arguments_.Read<std::string>();
    std::string cnstRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);

    maskMin[0] = OP{}.compareStrings(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());
    maskMax[0] = OP{}.compareStrings(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());

    std::cout << "Where evaluation filterStringConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterStringConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::FilterColConst()
{
    auto colName = arguments_.Read<std::string>();
    U cnst = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

    switch (OP::interval)
    {
    case CpuFilterInterval::OUTER:
    case CpuFilterInterval::INNER:
        maskMin[0] = cnst >= reinterpret_cast<T*>(std::get<0>(colValMin))[0] &&
                     cnst <= reinterpret_cast<T*>(std::get<0>(colValMax))[0];
        maskMax[0] = cnst >= reinterpret_cast<T*>(std::get<0>(colValMin))[0] &&
                     cnst <= reinterpret_cast<T*>(std::get<0>(colValMax))[0];
        break;

    case CpuFilterInterval::NONE:
    default:
        maskMin[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValMin))[0], cnst);
        maskMax[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValMax))[0], cnst);
        break;
    }

    std::cout << "Where evaluation filterColConstMin: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0]
              << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterColConstMax: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0]
              << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::FilterConstCol()
{
    T cnst = arguments_.Read<T>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<U>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

    switch (OP::interval)
    {
    case CpuFilterInterval::INNER:
    case CpuFilterInterval::OUTER:
        maskMin[0] = cnst >= reinterpret_cast<U*>(std::get<0>(colValMin))[0] &&
                     cnst <= reinterpret_cast<U*>(std::get<0>(colValMax))[0];
        maskMax[0] = cnst >= reinterpret_cast<U*>(std::get<0>(colValMin))[0] &&
                     cnst <= reinterpret_cast<U*>(std::get<0>(colValMax))[0];
        break;

    case CpuFilterInterval::NONE:
    default:
        maskMin[0] = OP{}.template operator()<T, U>(reinterpret_cast<U*>(std::get<0>(colValMin))[0], cnst);
        maskMax[0] = OP{}.template operator()<T, U>(reinterpret_cast<U*>(std::get<0>(colValMax))[0], cnst);
        break;
    }

    std::cout << "Where evaluation filterConstColMin: " << reinterpret_cast<U*>(std::get<0>(colValMin))[0]
              << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterConstColMax: " << reinterpret_cast<U*>(std::get<0>(colValMax))[0]
              << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (LoadCol<T>(colNameLeft) || LoadCol<U>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = GetPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = GetPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers_.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers_.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers_.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers_.at(colPointerNameRightMax);

        maskMin = AllocateRegister<int8_t>(reg + "_min", 1,
                                           std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1,
                                           std::get<2>(colValLeftMax) || std::get<2>(colValRightMax));

        maskMin[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                                    reinterpret_cast<U*>(std::get<0>(colValRightMin))[0]);
        maskMax[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMax))[0],
                                                    reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]);
    }

    std::cout << "Where evaluation filterColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterConstConst()
{
    T constLeft = arguments_.Read<T>();
    U constRight = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);

    maskMin[0] = OP{}.template operator()<T, U>(constLeft, constRight);
    maskMax[0] = OP{}.template operator()<T, U>(constLeft, constRight);

    std::cout << "Where evaluation filterConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::LogicalColConst()
{
    auto colName = arguments_.Read<std::string>();
    U cnst = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (LoadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    if (colName.front() != '$')
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    }
    else
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);
    }

    maskMin[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValMin))[0], cnst);
    maskMax[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValMax))[0], cnst);

    std::cout
        << "Where evaluation logicalColConstMin: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0]
        << ", " << reg + "_min"
        << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout
        << "Where evaluation logicalColConstMax: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0]
        << ", " << reg + "_max"
        << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::LogicalConstCol()
{
    T cnst = arguments_.Read<T>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (LoadCol<U>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    if (colName.front() != '$')
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    }
    else
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);
    }

    maskMin[0] = OP{}.template operator()<T, U>(cnst, reinterpret_cast<U*>(std::get<0>(colValMin))[0]);
    maskMax[0] = OP{}.template operator()<T, U>(cnst, reinterpret_cast<U*>(std::get<0>(colValMax))[0]);

    std::cout
        << "Where evaluation logicalConstColMin: " << reinterpret_cast<U*>(std::get<0>(colValMin))[0]
        << ", " << reg + "_min"
        << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout
        << "Where evaluation logicalConstColMax: " << reinterpret_cast<U*>(std::get<0>(colValMax))[0]
        << ", " << reg + "_max"
        << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::LogicalColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (LoadCol<T>(colNameLeft) || LoadCol<U>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = GetPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = GetPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers_.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers_.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers_.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers_.at(colPointerNameRightMax);

        if (colNameLeft.front() != '$' || colNameRight.front() != '$')
        {
            maskMin = AllocateRegister<int8_t>(reg + "_min", 1,
                                               std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
            maskMax = AllocateRegister<int8_t>(reg + "_max", 1,
                                               std::get<2>(colValLeftMax) || std::get<2>(colValRightMax));

            maskMin[0] =
                OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                               reinterpret_cast<U*>(std::get<0>(colValRightMin))[0]);
            maskMax[0] =
                OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMax))[0],
                                               reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]);
        }
        else
        {
            maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
            maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);

            maskMin[0] =
                OP{}.template operator()<T, U>(LogicOperations::logicalOr{}.template operator()<T, U>(
                                                   reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                                   reinterpret_cast<U*>(std::get<0>(colValLeftMax))[0]),
                                               LogicOperations::logicalOr{}.template operator()<T, U>(
                                                   reinterpret_cast<T*>(std::get<0>(colValRightMin))[0],
                                                   reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]));
            maskMax[0] =
                OP{}.template operator()<T, U>(LogicOperations::logicalOr{}.template operator()<T, U>(
                                                   reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                                   reinterpret_cast<U*>(std::get<0>(colValLeftMax))[0]),
                                               LogicOperations::logicalOr{}.template operator()<T, U>(
                                                   reinterpret_cast<T*>(std::get<0>(colValRightMin))[0],
                                                   reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]));
        }
    }

    std::cout << "Where evaluation logicalColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation logicalColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::LogicalConstConst()
{
    T constLeft = arguments_.Read<T>();
    U constRight = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, false);
    maskMin[0] = OP{}.template operator()<T, U>(constLeft, constRight);
    maskMax[0] = OP{}.template operator()<T, U>(constLeft, constRight);

    std::cout << "Where evaluation logicalConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation logicalConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;
    return 0;
}

template <typename T>
int32_t CpuSqlDispatcher::LogicalNotCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;

    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);
    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    int8_t* resultMin = AllocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* resultMax = AllocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    resultMin[0] = !reinterpret_cast<T*>(std::get<0>(colValMin))[0];
    resultMax[0] = !reinterpret_cast<T*>(std::get<0>(colValMax))[0];

    std::cout << "Where evaluation logicalNotCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation logicalNotCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;
    return 0;
}

template <typename T>
int32_t CpuSqlDispatcher::LogicalNotConst()
{
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    int8_t* resultMin = AllocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* resultMax = AllocateRegister<int8_t>(reg + "_max", 1, false);

    resultMin[0] = !cnst;
    resultMax[0] = !cnst;

    std::cout << "Where evaluation logicalNotConstMin: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation logicalNotConstMax: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}