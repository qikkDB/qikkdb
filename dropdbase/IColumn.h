#pragma once
#include "DataType.h"
#include <string>

class IColumn {
public:
	virtual const std::string& GetName() const = 0;
	virtual DataType GetColumnType() const = 0;

	virtual ~IColumn() {};
};