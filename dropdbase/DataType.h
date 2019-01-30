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
	CONST_BOOLEAN = 7,
	COLUMN_INT = 8,
	COLUMN_LONG = 9,
	COLUMN_FLOAT = 10,
	COLUMN_DOUBLE = 11,
	COLUMN_POINT = 12,
	COLUMN_POLYGON = 13,
	COLUMN_STRING = 14,
	COLUMN_BOOL = 15, //TODO either to represent it as int8_t or to remove bool column at all
	REG = 16,
	DATA_TYPE_SIZE = 17
};
