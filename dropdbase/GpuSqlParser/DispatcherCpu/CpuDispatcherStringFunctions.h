#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/StringOperations.h"

template <typename OP>
int32_t CpuSqlDispatcher::StringUnaryCol()
{
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

    std::string resultStringMin =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin));
    std::string resultStringMax =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax));

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
                                             std::get<2>(colValMax) || !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::StringUnaryConst()
{
    auto cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::string resultStringMin = OP{}(cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(cnst.c_str(), cnst.size());

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    return 0;
}


template <typename OP>
int32_t CpuSqlDispatcher::StringUnaryNumericCol()
{
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

    int32_t* resultMin = AllocateRegister<int32_t>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
    int32_t* resultMax = AllocateRegister<int32_t>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);

    resultMin[0] = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin));
    resultMax[0] = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax));

    return 0;
}

template <typename OP>
int32_t CpuSqlDispatcher::StringUnaryNumericConst()
{
    auto cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t* resultMin = AllocateRegister<int32_t>(reg + "_min", 1, !OP::isMonotonous);
    int32_t* resultMax = AllocateRegister<int32_t>(reg + "_max", 1, !OP::isMonotonous);

    resultMin[0] = OP{}(cnst.c_str(), cnst.size());
    resultMax[0] = OP{}(cnst.c_str(), cnst.size());

    return 0;
}

template <typename OP, typename T>
int32_t CpuSqlDispatcher::StringBinaryNumericColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    char* resultMin = nullptr;
    char* resultMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        resultMin = AllocateRegister<char>(reg + "_min", 1, true);
        resultMax = AllocateRegister<char>(reg + "_max", 1, true);
        resultMin[0] = 'a';
        resultMax[0] = 'a';
    }
    else
    {
        if (LoadCol<std::string>(colNameLeft) || LoadCol<T>(colNameRight))
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

        std::string resultStringMin =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMin)), std::get<1>(colValLeftMin),
                 reinterpret_cast<T*>(std::get<0>(colValRightMin))[0]);
        std::string resultStringMax =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMax)), std::get<1>(colValLeftMax),
                 reinterpret_cast<T*>(std::get<0>(colValRightMax))[0]);

        resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                           std::get<2>(colValLeftMin) ||
                                               std::get<2>(colValRightMin) || !OP::isMonotonous);
        resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryNumericColConst()
{
    auto colName = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
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

    std::string resultStringMin =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin), cnst);
    std::string resultStringMax =
        OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax), cnst);

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryNumericConstCol()
{
    std::string cnst = arguments_.Read<std::string>();
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

    std::string resultStringMin =
        OP{}(cnst.c_str(), cnst.size(), reinterpret_cast<T*>(std::get<0>(colValMin))[0]);
    std::string resultStringMax =
        OP{}(cnst.c_str(), cnst.size(), reinterpret_cast<T*>(std::get<0>(colValMax))[0]);

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryNumericConstConst()
{
    std::string cnstLeft = arguments_.Read<std::string>();
    T cnstRight = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    std::string resultStringMin = OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight);
    std::string resultStringMax = OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight);

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

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
int32_t CpuSqlDispatcher::StringBinaryColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    char* resultMin = nullptr;
    char* resultMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        resultMin = AllocateRegister<char>(reg + "_min", 1, true);
        resultMax = AllocateRegister<char>(reg + "_max", 1, true);
        resultMin[0] = 'a';
        resultMax[0] = 'a';
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

        std::string resultStringMin =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMin)), std::get<1>(colValLeftMin),
                 reinterpret_cast<char*>(std::get<0>(colValRightMin)), std::get<1>(colValRightMin));
        std::string resultStringMax =
            OP{}(reinterpret_cast<char*>(std::get<0>(colValLeftMax)), std::get<1>(colValLeftMax),
                 reinterpret_cast<char*>(std::get<0>(colValRightMax)), std::get<1>(colValRightMax));

        resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                           std::get<2>(colValLeftMin) ||
                                               std::get<2>(colValRightMin) || !OP::isMonotonous);
        resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryColConst()
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

    std::string resultStringMin = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                       std::get<1>(colValMin), cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                       std::get<1>(colValMax), cnst.c_str(), cnst.size());

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryConstCol()
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

    std::string resultStringMin = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)),
                                       std::get<1>(colValMin), cnst.c_str(), cnst.size());
    std::string resultStringMax = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)),
                                       std::get<1>(colValMax), cnst.c_str(), cnst.size());

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1,
                                             std::get<2>(colValMin) || !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1,
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
int32_t CpuSqlDispatcher::StringBinaryConstConst()
{
    std::string cnstLeft = arguments_.Read<std::string>();
    std::string cnstRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::string resultStringMin =
        OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());
    std::string resultStringMax =
        OP{}(cnstLeft.c_str(), cnstLeft.size(), cnstRight.c_str(), cnstRight.size());

    char* resultMin = AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1, !OP::isMonotonous);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1, !OP::isMonotonous);

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