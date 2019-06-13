#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"

/// Implementation of INSERT INTO operation
/// This executes once for every column-value pair
/// Insert value in column is referenced in the INSERT into command or null value if not
/// <returns name="statusCode">Finish status code of the operation</returns>
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