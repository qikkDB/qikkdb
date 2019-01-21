#pragma once
#include "DataType.h"

class IColumn {
public:
	virtual DataType GetColumnType() const = 0;

	virtual ~IColumn() {};
};