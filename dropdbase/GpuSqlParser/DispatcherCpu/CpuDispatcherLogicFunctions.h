#pragma once

#include "../CpuSqlDispatcher.h"
#include "CpuFilterInterval.h"
#include <tuple>

template <typename OP>
int32_t CpuSqlDispatcher::filterStringColConst()
{
    auto colName = arguments.read<std::string>();
    std::string cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    if (loadCol<std::string>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

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
int32_t CpuSqlDispatcher::filterStringConstCol()
{
    std::string cnst = arguments.read<std::string>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    if (loadCol<std::string>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

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
int32_t CpuSqlDispatcher::filterStringColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (loadCol<std::string>(colNameLeft) || loadCol<std::string>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = getPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = getPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers.at(colPointerNameRightMax);

        maskMin = allocateRegister<int8_t>(reg + "_min", 1,
                                           std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
        maskMax = allocateRegister<int8_t>(reg + "_max", 1,
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
int32_t CpuSqlDispatcher::filterStringConstConst()
{
    std::string cnstLeft = arguments.read<std::string>();
    std::string cnstRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);

    maskMin[0] = OP{}.compareStrings(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());
    maskMax[0] = OP{}.compareStrings(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());

    std::cout << "Where evaluation filterStringConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterStringConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterColConst()
{
    auto colName = arguments.read<std::string>();
    U cnst = arguments.read<U>();
    auto reg = arguments.read<std::string>();

    if (loadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

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
int32_t CpuSqlDispatcher::filterConstCol()
{
    T cnst = arguments.read<T>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    if (loadCol<U>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));

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
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (loadCol<T>(colNameLeft) || loadCol<U>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = getPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = getPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers.at(colPointerNameRightMax);

        maskMin = allocateRegister<int8_t>(reg + "_min", 1,
                                           std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
        maskMax = allocateRegister<int8_t>(reg + "_max", 1,
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
    T constLeft = arguments.read<T>();
    U constRight = arguments.read<U>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);

    maskMin[0] = OP{}.template operator()<T, U>(constLeft, constRight);
    maskMax[0] = OP{}.template operator()<T, U>(constLeft, constRight);

    std::cout << "Where evaluation filterConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation filterConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::logicalColConst()
{
    auto colName = arguments.read<std::string>();
    U cnst = arguments.read<U>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (loadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    if (colName.front() != '$')
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    }
    else
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);
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
int32_t CpuSqlDispatcher::logicalConstCol()
{
    T cnst = arguments.read<T>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (loadCol<U>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    if (colName.front() != '$')
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    }
    else
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);
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
int32_t CpuSqlDispatcher::logicalColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = nullptr;
    int8_t* maskMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
        maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);
        maskMin[0] = 1;
        maskMax[0] = 1;
    }
    else
    {
        if (loadCol<T>(colNameLeft) || loadCol<U>(colNameRight))
        {
            return 1;
        }

        std::string colPointerNameLeftMin;
        std::string colPointerNameLeftMax;
        std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = getPointerNames(colNameLeft);

        std::string colPointerNameRightMin;
        std::string colPointerNameRightMax;
        std::tie(colPointerNameRightMin, colPointerNameRightMax) = getPointerNames(colNameRight);

        auto colValLeftMin = allocatedPointers.at(colPointerNameLeftMin);
        auto colValLeftMax = allocatedPointers.at(colPointerNameLeftMax);

        auto colValRightMin = allocatedPointers.at(colPointerNameRightMin);
        auto colValRightMax = allocatedPointers.at(colPointerNameRightMax);

        if (colNameLeft.front() != '$' || colNameRight.front() != '$')
        {
            maskMin = allocateRegister<int8_t>(reg + "_min", 1,
                                               std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
            maskMax = allocateRegister<int8_t>(reg + "_max", 1,
                                               std::get<2>(colValLeftMax) || std::get<2>(colValRightMax));
        }
        else
        {
            maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
            maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);
        }

        maskMin[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                                    reinterpret_cast<U*>(std::get<0>(colValRightMin))[0]);
        maskMax[0] = OP{}.template operator()<T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMax))[0],
                                                    reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]);
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
int32_t CpuSqlDispatcher::logicalConstConst()
{
    T constLeft = arguments.read<T>();
    U constRight = arguments.read<U>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, false);
    maskMin[0] = OP{}.template operator()<T, U>(constLeft, constRight);
    maskMax[0] = OP{}.template operator()<T, U>(constLeft, constRight);

    std::cout << "Where evaluation logicalConstConst_min: " << reg + "_min"
              << ": " << static_cast<int32_t>(maskMin[0]) << std::endl;
    std::cout << "Where evaluation logicalConstConst_max: " << reg + "_max"
              << ": " << static_cast<int32_t>(maskMax[0]) << std::endl;
    return 0;
}

template <typename T>
int32_t CpuSqlDispatcher::logicalNotCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    if (loadCol<T>(colName))
    {
        return 1;
    }

    std::string colPointerNameMin;
    std::string colPointerNameMax;

    std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);
    auto colValMin = allocatedPointers.at(colPointerNameMin);
    auto colValMax = allocatedPointers.at(colPointerNameMax);

    int8_t* resultMin = allocateRegister<int8_t>(reg + "_min", 1, std::get<2>(colValMin));
    int8_t* resultMax = allocateRegister<int8_t>(reg + "_max", 1, std::get<2>(colValMax));
    resultMin[0] = !reinterpret_cast<T*>(std::get<0>(colValMin))[0];
    resultMax[0] = !reinterpret_cast<T*>(std::get<0>(colValMax))[0];

    std::cout << "Where evaluation logicalNotCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation logicalNotCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;
    return 0;
}

template <typename T>
int32_t CpuSqlDispatcher::logicalNotConst()
{
    T cnst = arguments.read<T>();
    auto reg = arguments.read<std::string>();

    int8_t* resultMin = allocateRegister<int8_t>(reg + "_min", 1, false);
    int8_t* resultMax = allocateRegister<int8_t>(reg + "_max", 1, false);

    resultMin[0] = !cnst;
    resultMax[0] = !cnst;

    std::cout << "Where evaluation logicalNotConstMin: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation logicalNotConstMax: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}