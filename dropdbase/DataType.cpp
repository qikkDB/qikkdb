#include "DataType.h"

DataType GetColumnDataTypeFromString(const std::string& dataType)
{
	std::string type = dataType;

	for (auto &c : type)
	{
		c = toupper(c);
	}

	if (type == "INT")
	{
		return DataType::COLUMN_INT;
	}
	else if (type == "LONG" || type == "DATE")
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
	else if (type == "POINT")
	{
		return DataType::COLUMN_POINT;
	}
	else if (type == "POLYGON")
	{
		return DataType::COLUMN_POLYGON;
	}
	else if (type == "STRING")
	{
		return DataType::COLUMN_STRING;
	}
	else if (type == "BOOLEAN")
	{
		return DataType::COLUMN_INT8_T;
	}
	else
	{
		return DataType::CONST_ERROR;
	}
}