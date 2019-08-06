#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"
#include "../InsertIntoStruct.h"

#ifndef __CUDACC__
template <typename T>
int32_t GpuSqlDispatcher::InsertInto()
{
    std::string column = arguments_.Read<std::string>();
    bool hasValue = arguments_.Read<bool>();

    T data = hasValue ? arguments_.Read<T>() : ColumnBase<T>::NullArray(1)[0];
    std::vector<T> dataVector({data});
    std::vector<int8_t> nullMaskVector({static_cast<int8_t>(hasValue ? 0 : 1)});

    insertIntoData_->insertIntoData.insert({column, dataVector});
    insertIntoNullMasks_.insert({column, nullMaskVector});
    return 0;
}
#endif
