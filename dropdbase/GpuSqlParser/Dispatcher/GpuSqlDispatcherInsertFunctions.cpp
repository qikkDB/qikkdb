#include "GpuSqlDispatcherInsertFunctions.h"
#include <array>
#include "../../PointFactory.h"
#include "../../ComplexPolygonFactory.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::insertIntoFunctions_ = {
    &GpuSqlDispatcher::InsertInto<int32_t>,
    &GpuSqlDispatcher::InsertInto<int64_t>,
    &GpuSqlDispatcher::InsertInto<float>,
    &GpuSqlDispatcher::InsertInto<double>,
    &GpuSqlDispatcher::InsertInto<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InsertInto<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InsertInto<std::string>,
    &GpuSqlDispatcher::InsertInto<int8_t>,
    &GpuSqlDispatcher::InsertInto<int32_t>,
    &GpuSqlDispatcher::InsertInto<int64_t>,
    &GpuSqlDispatcher::InsertInto<float>,
    &GpuSqlDispatcher::InsertInto<double>,
    &GpuSqlDispatcher::InsertInto<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InsertInto<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InsertInto<std::string>,
    &GpuSqlDispatcher::InsertInto<int8_t>};


template <>
int32_t GpuSqlDispatcher::InsertInto<ColmnarDB::Types::Point>()
{
    std::string column = arguments_.Read<std::string>();
    std::cout << "Column name: " << column << std::endl;
    bool hasValue = arguments_.Read<bool>();

    ColmnarDB::Types::Point point;

    if (hasValue)
    {
        std::string args = arguments_.Read<std::string>();
        std::cout << "Args: " << args << std::endl;
        point = PointFactory::FromWkt(args);
    }
    else
    {
        point = ColumnBase<ColmnarDB::Types::Point>::NullArray(1)[0];
    }
    std::vector<ColmnarDB::Types::Point> pointVector({point});
    std::vector<int8_t> nullMaskVector({static_cast<int8_t>(hasValue ? 0 : 1)});

    insertIntoData_->insertIntoData.insert({column, pointVector});
    insertIntoNullMasks_.insert({column, nullMaskVector});
    return 0;
}

template <>
int32_t GpuSqlDispatcher::InsertInto<ColmnarDB::Types::ComplexPolygon>()
{
    std::string column = arguments_.Read<std::string>();
    bool hasValue = arguments_.Read<bool>();

    ColmnarDB::Types::ComplexPolygon polygon;

    if (hasValue)
    {
        std::string args = arguments_.Read<std::string>();
        polygon = ComplexPolygonFactory::FromWkt(args);
    }
    else
    {
        polygon = ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(1)[0];
    }
    std::vector<ColmnarDB::Types::ComplexPolygon> polygonVector({polygon});
    std::vector<int8_t> nullMaskVector({static_cast<int8_t>(hasValue ? 0 : 1)});

    insertIntoData_->insertIntoData.insert({column, polygonVector});
    insertIntoNullMasks_.insert({column, nullMaskVector});

    return 0;
}

int32_t GpuSqlDispatcher::InsertIntoDone()
{
    std::string table = arguments_.Read<std::string>();
    database_->GetTables().at(table).InsertData(insertIntoData_->insertIntoData, false, insertIntoNullMasks_);
    insertIntoData_->insertIntoData.clear();
    insertIntoNullMasks_.clear();
    return 5;
}