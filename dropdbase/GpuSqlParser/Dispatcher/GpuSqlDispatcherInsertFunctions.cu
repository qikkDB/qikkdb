#include "GpuSqlDispatcherInsertFunctions.h"
#include <array>

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::insertIntoFunctions = { &GpuSqlDispatcher::insertInto<int32_t>, &GpuSqlDispatcher::insertInto<int64_t>, &GpuSqlDispatcher::insertInto<float>, &GpuSqlDispatcher::insertInto<double>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::insertInto<std::string>, &GpuSqlDispatcher::insertInto<int8_t>, &GpuSqlDispatcher::insertInto<int32_t>, &GpuSqlDispatcher::insertInto<int64_t>, &GpuSqlDispatcher::insertInto<float>, &GpuSqlDispatcher::insertInto<double>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::insertInto<std::string>, &GpuSqlDispatcher::insertInto<int8_t> };


template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>()
{
	std::string table = arguments.read<std::string>();
	std::string column = arguments.read<std::string>();
	bool isReferencedColumn = arguments.read<bool>();

	if (isReferencedColumn)
	{
		std::string args = arguments.read<std::string>();
		ColmnarDB::Types::Point point = PointFactory::FromWkt(args);

		dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertData({ point });
	}
	else
	{
		dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertNullData(1);
	}
	return 0;
}

template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>()
{
	std::string table = arguments.read<std::string>();
	std::string column = arguments.read<std::string>();
	bool isReferencedColumn = arguments.read<bool>();

	if (isReferencedColumn)
	{
		std::string args = arguments.read<std::string>();
		ColmnarDB::Types::ComplexPolygon polygon = ComplexPolygonFactory::FromWkt(args);

		dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertData({ polygon });
	}
	else
	{
		dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertNullData(1);
	}
	return 0;
}

int32_t GpuSqlDispatcher::insertIntoDone()
{
	return 5;
}