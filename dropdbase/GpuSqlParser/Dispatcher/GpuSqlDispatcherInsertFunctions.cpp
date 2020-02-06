#include "GpuSqlDispatcherInsertFunctions.h"
#include <array>
#include "../../PointFactory.h"
#include "../../ComplexPolygonFactory.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::insertIntoFunctions_)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, int32_t)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, int64_t)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, float)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, double)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, ColmnarDB::Types::Point)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, std::string)
DISPATCHER_UNARY_FUNCTION_NO_COL(GpuSqlDispatcher::InsertInto, int8_t)
END_DISPATCH_TABLE

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::InsertInto<ColmnarDB::Types::Point>()
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
    std::vector<int64_t> nullMaskVector({static_cast<int64_t>(hasValue ? 0 : 1)});

    insertIntoData_->insertIntoData.insert({column, pointVector});
    insertIntoNullMasks_.insert({column, nullMaskVector});
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::InsertInto<ColmnarDB::Types::ComplexPolygon>()
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
    std::vector<int64_t> nullMaskVector({static_cast<int64_t>(hasValue ? 0 : 1)});

    insertIntoData_->insertIntoData.insert({column, polygonVector});
    insertIntoNullMasks_.insert({column, nullMaskVector});

    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::InsertIntoDone()
{
    Context& context = Context::getInstance();

    std::string table = arguments_.Read<std::string>();

    for (auto& column : insertIntoData_->insertIntoData)
    {
        const int32_t blockCount =
            database_->GetTables().at(table).GetColumns().at(column.first)->GetBlockCount();
        const bool isColNullable =
            database_->GetTables().at(table).GetColumns().at(column.first)->GetIsNullable();

        if (database_->GetTables().at(table).GetSortingColumns().empty())
        {
            const int32_t lastBlockIdx = std::max(blockCount - 1, 0);

            context.getCacheForDevice(lastBlockIdx % context.getDeviceCount())
                .clearCachedBlock(database_->GetName(), table + "." + column.first, lastBlockIdx);

            if (isColNullable)
            {
                context.getCacheForDevice(lastBlockIdx % context.getDeviceCount())
                    .clearCachedBlock(database_->GetName(), table + "." + column.first + NULL_SUFFIX, lastBlockIdx);
            }
        }
        else
        {
            for (int32_t i = 0; i < blockCount; i++)
            {
                context.getCacheForDevice(i % context.getDeviceCount())
                    .clearCachedBlock(database_->GetName(), table + "." + column.first, i);

                if (isColNullable)
                {
                    context.getCacheForDevice(i % context.getDeviceCount())
                        .clearCachedBlock(database_->GetName(), table + "." + column.first + NULL_SUFFIX, i);
                }
            }
        }
    }

    database_->GetTables().at(table).InsertData(insertIntoData_->insertIntoData, false, insertIntoNullMasks_);
    insertIntoData_->insertIntoData.clear();
    insertIntoNullMasks_.clear();

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Insert into completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

void GpuSqlDispatcher::ClearArguments()
{
    arguments_.Clear();
}

void GpuSqlDispatcher::ClearInstructions()
{
    dispatcherFunctions_.clear();
}
