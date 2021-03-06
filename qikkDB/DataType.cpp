#include "DataType.h"

DataType GetColumnDataTypeFromString(const std::string& dataType)
{
    std::string type = dataType;

    for (auto& c : type)
    {
        c = toupper(c);
    }

    if (type == "INT" || type == "INTEGER" || type == "INT32")
    {
        return DataType::COLUMN_INT;
    }
    else if (type == "LONG" || type == "INT64" || type == "DATE" || type == "DATETIME")
    {
        return DataType::COLUMN_LONG;
    }
    else if (type == "FLOAT")
    {
        return DataType::COLUMN_FLOAT;
    }
    else if (type == "DOUBLE")
    {
        return DataType::COLUMN_DOUBLE;
    }
    else if (type == "GEO_POINT")
    {
        return DataType::COLUMN_POINT;
    }
    else if (type == "GEO_POLYGON")
    {
        return DataType::COLUMN_POLYGON;
    }
    else if (type == "STRING")
    {
        return DataType::COLUMN_STRING;
    }
    else if (type == "BOOLEAN" || type == "BOOL" || type == "INT8")
    {
        return DataType::COLUMN_INT8_T;
    }
    else
    {
        return DataType::CONST_ERROR;
    }
}

PayloadType GetPayloadTypeFromString(const std::string& dataType)
{
    std::string type = dataType;

    for (auto& c : type)
    {
        c = toupper(c);
    }
    if ( type == "DATE" || type == "DATETIME")
    {
        return PayloadType::PAYLOAD_DATE;
    }
    else
    {
        return PayloadType::PAYLOAD_DEFAULT;
    }
}

std::string GetStringFromColumnDataType(DataType type)
{
    switch (type)
    {
    case DataType::COLUMN_INT:
        return "INT";
        break;
    case DataType::COLUMN_LONG:
        return "LONG";
        break;
    case DataType::COLUMN_FLOAT:
        return "FLOAT";
        break;
    case DataType::COLUMN_DOUBLE:
        return "DOUBLE";
        break;
    case DataType::COLUMN_POINT:
        return "POINT";
        break;
    case DataType::COLUMN_POLYGON:
        return "POLYGON";
        break;
    case DataType::COLUMN_STRING:
        return "STRING";
        break;
    case DataType::COLUMN_INT8_T:
        return "BOOL";
        break;
    default:
        return "";
        break;
    }
}

DataType GetConstDataType(DataType type)
{
    if (type >= numOfDataTypes)
    {
        return static_cast<DataType>(type - numOfDataTypes);
    }
    return type;
}