#pragma once

#include "DataType.h"

class IVariantArray
{
public:
    virtual DataType GetType() const = 0;
    virtual int64_t GetSize() const = 0;

    virtual ~IVariantArray(){};
};