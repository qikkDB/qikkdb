#pragma once

#include "../CpuSqlDispatcher.h"
#include <tuple>

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::ArithmeticColConst()
{
    auto colName = arguments_.Read<std::string>();
    U cnst = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    constexpr bool bothTypesFloatOrBothIntegral =
        (std::is_floating_point<T>::value && std::is_floating_point<U>::value) ||
        (std::is_integral<T>::value && std::is_integral<U>::value);
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;

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

    resultMin[0] =
        OP{}.template operator()<ResultType, T, U>(reinterpret_cast<T*>(std::get<0>(colValMin))[0], cnst);
    resultMax[0] =
        OP{}.template operator()<ResultType, T, U>(reinterpret_cast<T*>(std::get<0>(colValMax))[0], cnst);

    std::cout << std::string(typeid(ResultType).name()) << std::endl;
    std::cout << "Where evaluation arithmeticColConstMin: "
              << reinterpret_cast<T*>(std::get<0>(colValMin))[0] << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation arithmeticColConstMax: "
              << reinterpret_cast<T*>(std::get<0>(colValMax))[0] << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticConstCol()
{
    T cnst = arguments_.Read<T>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    constexpr bool bothTypesFloatOrBothIntegral =
        (std::is_floating_point<T>::value && std::is_floating_point<U>::value) ||
        (std::is_integral<T>::value && std::is_integral<U>::value);
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;

    if (LoadCol<U>(colName))
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

    resultMin[0] =
        OP{}.template operator()<ResultType, T, U>(cnst, reinterpret_cast<U*>(std::get<0>(colValMin))[0]);
    resultMax[0] =
        OP{}.template operator()<ResultType, T, U>(cnst, reinterpret_cast<U*>(std::get<0>(colValMax))[0]);

    std::cout << "Where evaluation arithmeticConstColMin: "
              << reinterpret_cast<T*>(std::get<0>(colValMin))[0] << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation arithmeticConstColMax: "
              << reinterpret_cast<T*>(std::get<0>(colValMax))[0] << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();
    constexpr bool bothTypesFloatOrBothIntegral =
        (std::is_floating_point<T>::value && std::is_floating_point<U>::value) ||
        (std::is_integral<T>::value && std::is_integral<U>::value);
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;

    ResultType* resultMin = nullptr;
    ResultType* resultMax = nullptr;

    if (colNameLeft.front() != '$' && colNameRight.front() != '$')
    {
        resultMin = AllocateRegister<ResultType>(reg + "_min", 1, true);
        resultMax = AllocateRegister<ResultType>(reg + "_max", 1, true);
        resultMin[0] = 1;
        resultMax[0] = 1;
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

        resultMin = AllocateRegister<ResultType>(reg + "_min", 1,
                                                 std::get<2>(colValLeftMin) ||
                                                     std::get<2>(colValRightMin) || !OP::isMonotonous);
        resultMax = AllocateRegister<ResultType>(reg + "_max", 1,
                                                 std::get<2>(colValLeftMax) ||
                                                     std::get<2>(colValRightMax) || !OP::isMonotonous);

        resultMin[0] =
            OP{}.template operator()<ResultType, T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0],
                                                       reinterpret_cast<U*>(std::get<0>(colValRightMin))[0]);
        resultMax[0] =
            OP{}.template operator()<ResultType, T, U>(reinterpret_cast<T*>(std::get<0>(colValLeftMax))[0],
                                                       reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]);
    }
    std::cout << "Where evaluation arithmeticColCol_min: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation arithmeticColCol_max: " << colNameLeft << ", " << colNameRight
              << ", " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}

template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticConstConst()
{
    T constLeft = arguments_.Read<T>();
    U constRight = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();
    constexpr bool bothTypesFloatOrBothIntegral =
        (std::is_floating_point<T>::value && std::is_floating_point<U>::value) ||
        (std::is_integral<T>::value && std::is_integral<U>::value);
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral, typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type>::type ResultType;

    ResultType* resultMin = AllocateRegister<ResultType>(reg + "_min", 1, false);
    ResultType* resultMax = AllocateRegister<ResultType>(reg + "_max", 1, false);

    resultMin[0] = OP{}.template operator()<ResultType, T, U>(constLeft, constRight);
    resultMax[0] = OP{}.template operator()<ResultType, T, U>(constLeft, constRight);

    std::cout << "Where evaluation arithmeticConstConst_min: " << reg + "_min"
              << ": " << resultMin[0] << std::endl;
    std::cout << "Where evaluation arithmeticConstConst_max: " << reg + "_max"
              << ": " << resultMax[0] << std::endl;

    return 0;
}