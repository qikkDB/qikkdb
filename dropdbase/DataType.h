#pragma once
#include <cstdint>
#include "Types/Point.pb.h"
#include "Types/ComplexPolygon.pb.h"


enum DataType
{
    CONST_ERROR = -1,
    CONST_INT = 0,
    CONST_LONG = 1,
    CONST_FLOAT = 2,
    CONST_DOUBLE = 3,
    CONST_POINT = 4,
    CONST_POLYGON = 5,
    CONST_STRING = 6,
    CONST_INT8_T = 7,
    COLUMN_INT = 8,
    COLUMN_LONG = 9,
    COLUMN_FLOAT = 10,
    COLUMN_DOUBLE = 11,
    COLUMN_POINT = 12,
    COLUMN_POLYGON = 13,
    COLUMN_STRING = 14,
    COLUMN_INT8_T = 15,
    DATA_TYPE_SIZE = 16
};

constexpr int32_t numOfDataTypes = DATA_TYPE_SIZE / 2;

constexpr int32_t GetDataTypeSize(DataType type)
{
    switch (type)
    {
    case COLUMN_INT:
    case CONST_INT:
        return sizeof(int32_t);
    case COLUMN_LONG:
    case CONST_LONG:
        return sizeof(int64_t);
    case COLUMN_DOUBLE:
    case CONST_DOUBLE:
        return sizeof(double);
    case COLUMN_FLOAT:
    case CONST_FLOAT:
        return sizeof(float);
    default:
        return sizeof(int8_t);
    }
}

template <typename T>
constexpr DataType GetColumnType()
{
    typedef typename std::conditional<
        std::is_same<T, int>::value, std::integral_constant<DataType, COLUMN_INT>,
        typename std::conditional<
            std::is_same<T, int64_t>::value, std::integral_constant<DataType, COLUMN_LONG>,
            typename std::conditional<
                std::is_same<T, float>::value, std::integral_constant<DataType, COLUMN_FLOAT>,
                typename std::conditional<
                    std::is_same<T, double>::value, std::integral_constant<DataType, COLUMN_DOUBLE>,
                    typename std::conditional<
                        std::is_same<T, ColmnarDB::Types::Point>::value, std::integral_constant<DataType, COLUMN_POINT>,
                        typename std::conditional<
                            std::is_same<T, ColmnarDB::Types::ComplexPolygon>::value, std::integral_constant<DataType, COLUMN_POLYGON>,
                            typename std::conditional<std::is_same<T, std::string>::value, std::integral_constant<DataType, COLUMN_STRING>,
                                                      typename std::conditional<std::is_same<T, bool>::value, std::integral_constant<DataType, COLUMN_INT8_T>,
                                                                                typename std::conditional<std::is_same<T, int8_t>::value, std::integral_constant<DataType, COLUMN_INT8_T>, std::integral_constant<DataType, CONST_ERROR>>::type>::
                                                          type>::type>::type>::type>::type>::type>::type>::type retConst;
    return retConst::value;
}

DataType GetColumnDataTypeFromString(const std::string& dataType);

std::string GetStringFromColumnDataType(DataType type);

DataType GetConstDataTypeFromColumn(DataType type);