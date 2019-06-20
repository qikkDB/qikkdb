#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"


#ifndef __CUDACC__
template<typename T>
int32_t GpuSqlDispatcher::insertInto()
{
	std::string column = arguments.read<std::string>();
	bool isReferencedColumn = arguments.read<bool>();

	T data = isReferencedColumn ? arguments.read<T>() : ColumnBase<T>::NullArray(1)[0];
	std::vector<T> dataVector({ data });

	insertIntoData.insert({ column, dataVector });
	return 0;
}
#endif

