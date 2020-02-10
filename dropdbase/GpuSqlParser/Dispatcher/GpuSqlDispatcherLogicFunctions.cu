#include "GpuSqlDispatcherLogicFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#define MERGED
#include "DispatcherMacros.h"

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::greaterFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::greater, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::greater, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greater, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::lessFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::less, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::less, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::less, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::greaterEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::greaterEqual, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::greaterEqual, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::greaterEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::lessEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::lessEqual, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::lessEqual, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::lessEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::equalFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::equal, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::equal, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::equal, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::notEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::notEqual, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::notEqual, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::notEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logicalAndFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalAnd, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalAnd, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalAnd, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalAnd, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, std::string)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalAnd, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logicalOrFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalOr, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalOr, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalOr, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalOr, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, std::string)
DISPATCHER_TYPE(GpuSqlDispatcher::Filter, FilterConditions::logicalOr, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::logicalNotFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::LogicalNot, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::LogicalNot, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::LogicalNot, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::LogicalNot, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::LogicalNot, int8_t)
END_DISPATCH_TABLE

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::isNullFunction_ =
    &GpuSqlDispatcher::NullMaskCol<NullMaskOperations::isNull>;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::isNotNullFunction_ =
    &GpuSqlDispatcher::NullMaskCol<NullMaskOperations::isNotNull>;


#undef MERGED