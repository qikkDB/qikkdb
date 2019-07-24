#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"
#include "../InsertIntoStruct.h"

#ifndef __CUDACC__
template<typename T>
int32_t GpuSqlDispatcher::insertInto()
{
	std::string column = arguments.read<std::string>();
	bool hasValue = arguments.read<bool>();

	T data = hasValue ? arguments.read<T>() : ColumnBase<T>::NullArray(1)[0];
	std::vector<T> dataVector({ data });
	std::vector<int8_t> nullMaskVector({ static_cast<int8_t>(hasValue ? 0 : 1) });

	insertIntoData->insertIntoData.insert({ column, dataVector });
	insertIntoNullMasks.insert({ column, nullMaskVector });
	return 0;
}
#endif

