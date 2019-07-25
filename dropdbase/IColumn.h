#pragma once
#include "DataType.h"
#include <utility>
#include <string>

class IColumn {
public:
	virtual const std::string& GetName() const = 0;
	virtual DataType GetColumnType() const = 0;
	virtual int32_t GetBlockCount() const = 0;
	virtual int64_t GetSize() const = 0;
	virtual void InsertNullData(int length) = 0;
	virtual float GetInitAvg() const = 0;
	virtual bool GetInitAvgIsSet() const = 0;
	virtual std::pair<int8_t*, size_t> GetNullBitMaskForBlock(size_t blockIndex) = 0;
	virtual bool GetIsNullable() const = 0;
	virtual void SetIsNullable(bool isNullable) = 0;

	virtual ~IColumn() {};
};