#pragma once
#include "../GpuSqlDispatcher.h"

template<typename T>
int32_t GpuSqlDispatcher::insertInto()
{
	std::string table = arguments.read<std::string>();
	std::string column = arguments.read<std::string>();
	bool isReferencedColumn = arguments.read<bool>();

	if (isReferencedColumn)
	{
		T args = arguments.read<T>();

		dynamic_cast<ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertData({ args });
	}
	else
	{
		dynamic_cast<ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertNullData(1);
	}
	return 0;
}