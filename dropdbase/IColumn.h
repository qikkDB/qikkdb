#pragma once
#include "DataType.h"
#include <string>

class IColumn {
public:
	virtual const std::string& GetName() const = 0;
	virtual DataType GetColumnType() const = 0;
	virtual int32_t GetBlockCount() const = 0;
	virtual const float GetInitAvg() const = 0;
	virtual const bool GetInitAvgIsSet() const = 0;

	virtual ~IColumn() {};
};