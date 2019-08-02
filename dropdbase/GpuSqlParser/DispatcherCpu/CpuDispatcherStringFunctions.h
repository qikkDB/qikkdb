#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/StringOperations.h"

template <typename OP>
int32_t CpuSqlDispatcher::stringUnaryCol()
{
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

    std::string resultStringMin =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin));
    std::string resultStringMax =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax));

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::stringUnaryConst()
{
    auto cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::string resultStringMin = OP{}(cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(cnst.c_str(), cnst.size());

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::stringUnaryNumericCol()
{
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

    int32_t* resultMin = allocateRegister<int32_t>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
    int32_t* resultMax = allocateRegister<int32_t>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);

    resultMin[0] = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin));
    resultMax[0] = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax));

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::stringUnaryNumericConst()
{
    auto cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t* resultMin = allocateRegister<int32_t>(reg + "_min", 1, !OP::isMonotonous);
    int32_t* resultMax = allocateRegister<int32_t>(reg + "_max", 1, !OP::isMonotonous);

    resultMin[0] = OP{}(cnst.c_str(), cnst.size());
    resultMax[0] = OP{}(cnst.c_str(), cnst.size());

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    char* resultMin = nullptr;
    char* resultMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        resultMin = allocateRegister<char>(reg + "_min", 1, true);
        resultMax = allocateRegister<char>(reg + "_max", 1, true);
        resultMin[0] = 'a';
        resultMax[0] = 'a';
    }
    else
    {
        if (loadCol<std::string>(colNameLeft) || loadCol<T>(colNameRight))
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

        std::string resultStringMin =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMin)), std::get<1>(colValLeftMin),
                 reinterpret_cast<T*>(std::get<0>(colValRightMin))[0]);
        std::string resultStringMax =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMax)), std::get<1>(colValLeftMax),
                 reinterpret_cast<T*>(std::get<0>(colValRightMax))[0]);

        resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                           std::get<2>(colValLeftMin) ||
                                               std::get<2>(colValRightMin) || !OP::isMonotonous);
        resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                           std::get<2>(colValLeftMax) ||
                                               std::get<2>(colValRightMax) || !OP::isMonotonous);

        std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
        resultMin[resultStringMin.size()] = '\0';
        std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
        resultMax[resultStringMax.size()] = '\0';
    }
    std::cout << "Where evaluation stringColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericColConst()
{
    auto colName = arguments.read<std::string>();
    T cnst = arguments.read<T>();
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

    std::string resultStringMin =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin), cnst);
    std::string resultStringMax =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax), cnst);

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringColConst_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringColConst_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericConstCol()
{
    std::string cnst = arguments.read<std::string>();
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

    std::string resultStringMin =
        OP{}(cnst.c_str(), cnst.size(), reinterpret_cast<T*>(std::get<0>(colValMin))[0]);
    std::string resultStringMax =
        OP{}(cnst.c_str(), cnst.size(), reinterpret_cast<T*>(std::get<0>(colValMax))[0]);

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringConstCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringConstCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericConstConst()
{
    std::string cnstLeft = arguments.read<std::string>();
    T cnstRight = arguments.read<T>();
    auto reg = arguments.read<std::string>();

    std::string resultStringMin = OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight);
    std::string resultStringMax = OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight);

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringConstConst_min: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringConstConst_max: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::stringBinaryColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    char* resultMin = nullptr;
    char* resultMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        resultMin = allocateRegister<char>(reg + "_min", 1, true);
        resultMax = allocateRegister<char>(reg + "_max", 1, true);
        resultMin[0] = 'a';
        resultMax[0] = 'a';
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

        std::string resultStringMin =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMin)), std::get<1>(colValLeftMin),
                 reinterpret_cast<char*>(std::get<0>(colValRightMin)), std::get<1>(colValRightMin));
        std::string resultStringMax =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMax)), std::get<1>(colValLeftMax),
                 reinterpret_cast<char*>(std::get<0>(colValRightMax)), std::get<1>(colValRightMax));

        resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                           std::get<2>(colValLeftMin) ||
                                               std::get<2>(colValRightMin) || !OP::isMonotonous);
        resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                           std::get<2>(colValLeftMax) ||
                                               std::get<2>(colValRightMax) || !OP::isMonotonous);

        std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
        resultMin[resultStringMin.size()] = '\0';
        std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
        resultMax[resultStringMax.size()] = '\0';
    }
    std::cout << "Where evaluation stringColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::stringBinaryColConst()
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

    std::string resultStringMin = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                       std::get<1>(colValMin), cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                       std::get<1>(colValMax), cnst.c_str(), cnst.size());

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringColConst_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringColConst_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::stringBinaryConstCol()
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

    std::string resultStringMin = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                       std::get<1>(colValMin), cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                       std::get<1>(colValMax), cnst.c_str(), cnst.size());

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringConstCol_min: " << colName << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringConstCol_max: " << colName << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::stringBinaryConstConst()
{
    std::string cnstLeft = arguments.read<std::string>();
    std::string cnstRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::string resultStringMin =
        OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());
    std::string resultStringMax =
        OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());

    char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    std::cout << "Where evaluation stringConstConst_min: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation stringConstConst_max: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}