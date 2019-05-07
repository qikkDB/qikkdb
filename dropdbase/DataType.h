#pragma once

enum DataType {
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


inline DataType getConstDataType(DataType operand)
{
	return operand >= DataType::COLUMN_INT ? static_cast<DataType>(operand - DataType::COLUMN_INT) : operand;
}
