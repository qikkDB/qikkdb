#include "GpuSqlDispatcherInsertFunctions.h"
#include <array>
#include "../../PointFactory.h"
#include "../../ComplexPolygonFactory.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::insertIntoFunctions = { &GpuSqlDispatcher::insertInto<int32_t>, &GpuSqlDispatcher::insertInto<int64_t>, &GpuSqlDispatcher::insertInto<float>, &GpuSqlDispatcher::insertInto<double>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::insertInto<std::string>, &GpuSqlDispatcher::insertInto<int8_t>, &GpuSqlDispatcher::insertInto<int32_t>, &GpuSqlDispatcher::insertInto<int64_t>, &GpuSqlDispatcher::insertInto<float>, &GpuSqlDispatcher::insertInto<double>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>, &GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::insertInto<std::string>, &GpuSqlDispatcher::insertInto<int8_t> };


template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>()
{
	std::string column = arguments.read<std::string>();
	bool hasValue = arguments.read<bool>();

	ColmnarDB::Types::Point point;

	if (hasValue)
	{
		std::string args = arguments.read<std::string>();
		point = PointFactory::FromWkt(args);
	}
	else
	{
		point = ColumnBase<ColmnarDB::Types::Point>::NullArray(1)[0];
	}
	std::vector<ColmnarDB::Types::Point> pointVector({ point });
	std::vector<int8_t> nullMaskVector({ static_cast<int8_t>(hasValue ? 0 : 1) });

	insertIntoData->insertIntoData.insert({ column, pointVector });
	insertIntoNullMasks.insert({ column, nullMaskVector });
	return 0;
}

template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>()
{
	std::string column = arguments.read<std::string>();
	bool hasValue = arguments.read<bool>();

	ColmnarDB::Types::ComplexPolygon polygon;

	if (hasValue)
	{
		std::string args = arguments.read<std::string>();
		polygon = ComplexPolygonFactory::FromWkt(args);
	}
	else
	{
		polygon = ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(1)[0];
	}
	std::vector<ColmnarDB::Types::ComplexPolygon> polygonVector({ polygon });
	std::vector<int8_t> nullMaskVector({ static_cast<int8_t>(hasValue ? 0 : 1) });
	
	insertIntoData->insertIntoData.insert({ column, polygonVector });
	insertIntoNullMasks.insert({ column, nullMaskVector });

	return 0;
}

int32_t GpuSqlDispatcher::insertIntoDone()
{
	std::string table = arguments.read<std::string>();
	database->GetTables().at(table).InsertData(insertIntoData->insertIntoData, false, insertIntoNullMasks);
	insertIntoData->insertIntoData.clear();
	insertIntoNullMasks.clear();
	return 5;
}