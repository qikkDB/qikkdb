#pragma once

#include <utility>
#include <string>

#include "DataType.h"
#include "BlockBase.h"

class IColumn
{
public:
    virtual const std::string& GetName() const = 0;
    virtual DataType GetColumnType() const = 0;
    virtual int32_t GetBlockCount() const = 0;
    virtual size_t GetBlockSize(int32_t blockIndex) const = 0;
    virtual int64_t GetSize() const = 0;
    virtual void UpdateSize() = 0;
    virtual int64_t GetBlockSizeForIndex(int32_t blockIdx) const = 0;
    virtual void InsertNullData(int length) = 0;
    virtual float GetInitAvg() const = 0;
    virtual bool GetInitAvgIsSet() const = 0;
    virtual std::pair<nullmask_t*, size_t> GetNullBitMaskForBlock(size_t blockIndex) = 0;
    virtual bool GetIsNullable() const = 0;
    virtual void SetIsNullable(bool isNullable) = 0;
    virtual bool GetIsUnique() const = 0;
    virtual void SetIsUnique(bool isUnique) = 0;
    virtual void SetColumnName(std::string newName) = 0;
    virtual void ResizeColumn(IColumn* srcColumnArg) = 0;
    virtual const std::string& GetFileAddressPath() const = 0;
    virtual const std::string& GetFileDataPath() const = 0;
    virtual const std::string& GetFileFragmentPath() const = 0;
    virtual const std::string& GetEncoding() const = 0;
    virtual void SetFileAddressPath(const std::string newFilePath) = 0;
    virtual void SetFileDataPath(const std::string newFilePath) = 0;
    virtual void SetFileFragmentPath(const std::string newFilePath) = 0;
    virtual void SetEncoding(const std::string newEncoding) = 0;

    virtual ~IColumn(){};
};