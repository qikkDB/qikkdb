#include "GpuSqlDispatcherAggregationFunctions.h"
#include <array>
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::minAggregationFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::min, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::min, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::min, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::min, double, double)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::maxAggregationFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::max, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::max, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::max, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::max, double, double)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::sumAggregationFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::sum, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::sum, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::sum, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::sum, double, double)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::countAggregationFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, double)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::count, int64_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::avgAggregationFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::avg, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::avg, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::avg, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Aggregation, AggregationFunctions::avg, double, double)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::minGroupByFunctions_)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::min, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(AggregationFunctions::min, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, std::string, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::min, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::maxGroupByFunctions_)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::max, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(AggregationFunctions::max, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, std::string, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::max, int8_t)
END_DISPATCH_TABLE


BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::sumGroupByFunctions_)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::sum, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(AggregationFunctions::sum, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, std::string, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::sum, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::countGroupByFunctions_)
DISPATCHER_GROUPBY_TYPE_WITH_RET(GpuSqlDispatcher::AggregationGroupBy,
                                 AggregationFunctions::count,
                                 int64_t,
                                 int32_t,
                                 1,
                                 1,
                                 1,
                                 1,
                                 0,
                                 0,
                                 0,
                                 0)
DISPATCHER_GROUPBY_TYPE_WITH_RET(GpuSqlDispatcher::AggregationGroupBy,
                                 AggregationFunctions::count,
                                 int64_t,
                                 int64_t,
                                 1,
                                 1,
                                 1,
                                 1,
                                 0,
                                 0,
                                 0,
                                 0)
DISPATCHER_GROUPBY_TYPE_WITH_RET(GpuSqlDispatcher::AggregationGroupBy,
                                 AggregationFunctions::count,
                                 int64_t,
                                 float,
                                 1,
                                 1,
                                 1,
                                 1,
                                 0,
                                 0,
                                 0,
                                 0)
DISPATCHER_GROUPBY_TYPE_WITH_RET(GpuSqlDispatcher::AggregationGroupBy,
                                 AggregationFunctions::count,
                                 int64_t,
                                 double,
                                 1,
                                 1,
                                 1,
                                 1,
                                 0,
                                 0,
                                 0,
                                 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::count, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(AggregationFunctions::count, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_GROUPBY_TYPE_WITH_RET(GpuSqlDispatcher::AggregationGroupBy,
                                 AggregationFunctions::count,
                                 int64_t,
                                 std::string,
                                 1,
                                 1,
                                 1,
                                 1,
                                 0,
                                 0,
                                 0,
                                 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::count, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::avgGroupByFunctions_)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::avg, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(AggregationFunctions::avg, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_GROUPBY_TYPE(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, std::string, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(AggregationFunctions::avg, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::minGroupByMultiKeyFunctions_)
DISPATCHER_ERR(Const, AggregationFunctions::min, std::vector<void*>)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, int32_t, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, AggregationFunctions::min, std::vector<void*>)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, int64_t, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, AggregationFunctions::min, std::vector<void*>)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, float, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, AggregationFunctions::min, std::vector<void*>)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::min, double, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::min, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::maxGroupByMultiKeyFunctions_)
DISPATCHER_ERR(ColConst, AggregationFunctions::max, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, int32_t, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::max, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, int64_t, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::max, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, float, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::max, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::max, double, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::max, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::sumGroupByMultiKeyFunctions_)
DISPATCHER_ERR(ColConst, AggregationFunctions::sum, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, int32_t, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::sum, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, int64_t, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::sum, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, float, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::sum, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::sum, double, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::sum, int8_t)
END_DISPATCH_TABLE


BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::countGroupByMultiKeyFunctions_)
DISPATCHER_ERR(ColConst, AggregationFunctions::count, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::count, int64_t, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::count, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::count, int64_t, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::count, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::count, int64_t, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::count, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::count, int64_t, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::count, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::avgGroupByMultiKeyFunctions_)
DISPATCHER_ERR(ColConst, AggregationFunctions::avg, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, int32_t, std::vector<void*>, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::avg, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, int64_t, std::vector<void*>, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::avg, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, float, std::vector<void*>, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(ColConst, AggregationFunctions::avg, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::AggregationGroupBy, AggregationFunctions::avg, double, std::vector<void*>, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, std::string)
DISPATCHER_UNARY_ERROR(AggregationFunctions::avg, int8_t)
END_DISPATCH_TABLE


BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::groupByFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::GroupBy, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::GroupBy, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::GroupBy, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::GroupBy, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::GroupBy, std::string)
DISPATCHER_UNARY_ERROR(int8_t)
END_DISPATCH_TABLE

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::groupByBeginFunction_ = &GpuSqlDispatcher::GroupByBegin;

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE + 1> GpuSqlDispatcher::groupByDoneFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<AggregationFunctions::none, int8_t>,
    &GpuSqlDispatcher::GroupByDone<int32_t>,
    &GpuSqlDispatcher::GroupByDone<int64_t>,
    &GpuSqlDispatcher::GroupByDone<float>,
    &GpuSqlDispatcher::GroupByDone<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<AggregationFunctions::none, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<AggregationFunctions::none, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::GroupByDone<std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<AggregationFunctions::none, int8_t>,
    &GpuSqlDispatcher::GroupByDone<std::vector<void*>>};

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::aggregationBeginFunction_ = &GpuSqlDispatcher::AggregationBegin;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::aggregationDoneFunction_ = &GpuSqlDispatcher::AggregationDone;

template <>
int32_t GpuSqlDispatcher::GroupByCol<std::string>()
{
    std::string columnName = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<std::string>(columnName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "GroupByString: " << columnName << '\n';

    const auto column = FindStringColumn(columnName); // Just copy!

    int32_t reconstructOutSize;
    GPUMemory::GPUString reconstructOutReg;
    int8_t* reconstructOutNullMask;
    GPUReconstruct::ReconstructStringColKeep(&reconstructOutReg, &reconstructOutSize, std::get<0>(column),
                                             reinterpret_cast<int8_t*>(filter_), std::get<1>(column),
                                             &reconstructOutNullMask, std::get<2>(column));

    FillStringRegister(reconstructOutReg, columnName + RECONSTRUCTED_SUFFIX, reconstructOutSize,
                       filter_ ? false : true, reconstructOutNullMask);
    InsertRegister(columnName + NULL_SUFFIX + RECONSTRUCTED_SUFFIX,
                   PointerAllocation{reinterpret_cast<uintptr_t>(reconstructOutNullMask),
                                     reconstructOutSize, filter_ ? true : false, 0});

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(columnName)) ==
        groupByColumns_.end())
    {
        groupByColumns_.push_back({columnName, DataType::COLUMN_STRING});
    }
    usingGroupBy_ = true;

    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::GroupByDone()
{
    bool containsAggFunction = arguments_.Read<bool>();
    insideGroupBy_ = false;

    // Preparation for group by without aggregation
    if (!containsAggFunction)
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "Group By without aggregation function: " << typeid(T).name() << '\n';

        if (groupByTables_[dispatcherThreadId_] == nullptr)
        {
            groupByTables_[dispatcherThreadId_] =
                GpuSqlDispatcher::GroupByHelper<AggregationFunctions::none, int32_t, T, int32_t>::CreateInstance(
                    Configuration::GetInstance().GetGroupByBuckets(), groupByColumns_);
        }

        GpuSqlDispatcher::GroupByHelper<AggregationFunctions::none, int32_t, T, int32_t>::ProcessBlock(
            groupByColumns_, PointerAllocation{0, std::numeric_limits<int32_t>::max(), false, 0}, *this);

        if (isLastBlockOfDevice_)
        {
            if (isOverallLastBlock_)
            {
                // Wait until all threads finished work
                std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
                GpuSqlDispatcher::groupByCV_.wait(lock,
                                                  [] { return GpuSqlDispatcher::IsGroupByDone(); });
                if (GpuSqlDispatcher::thrownException_)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::warning)
                        << "Skip reconstruction group by in thread: " << dispatcherThreadId_ << '\n';
                    return 13;
                }

                CudaLogBoost::getInstance(CudaLogBoost::debug)
                    << "Reconstructing group by in thread: " << dispatcherThreadId_ << '\n';

                std::string dummyRegName;
                GpuSqlDispatcher::GroupByHelper<AggregationFunctions::none, int32_t, T, int32_t>::GetResults(
                    groupByColumns_, dummyRegName, *this, false);
            }
            else
            {
                CudaLogBoost::getInstance(CudaLogBoost::debug)
                    << "Group by all blocks done in thread: " << dispatcherThreadId_ << '\n';
                // Increment counter and notify threads
                std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
                GpuSqlDispatcher::IncGroupByDoneCounter();
                GpuSqlDispatcher::groupByCV_.notify_all();
            }
        }
    }

    return 0;
}

int32_t GpuSqlDispatcher::GroupByBegin()
{
    usingGroupBy_ = true;
    insideGroupBy_ = true;
    return 0;
}

int32_t GpuSqlDispatcher::AggregationDone()
{
    insideAggregation_ = false;
    return 0;
}

int32_t GpuSqlDispatcher::AggregationBegin()
{
    insideAggregation_ = true;
    return 0;
}
