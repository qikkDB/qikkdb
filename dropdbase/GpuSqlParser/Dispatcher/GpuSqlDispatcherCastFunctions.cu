#include "GpuSqlDispatcherCastFunctions.h"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include <array>
#include "DispatcherMacros.h"
BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToIntFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, double)
DISPATCHER_UNARY_ERROR(int32_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int32_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToLongFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, double)
DISPATCHER_UNARY_ERROR(int64_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int64_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToFloatFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, double)
DISPATCHER_UNARY_ERROR(float, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(float, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToDoubleFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, double)
DISPATCHER_UNARY_ERROR(double, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(double, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, double)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToPointFunctions_)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int32_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int64_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, float)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, NativeGeoPoint)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToPolygonFunctions_)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int32_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int64_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, float)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, std::string)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToInt8TFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, double)
DISPATCHER_UNARY_ERROR(int8_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int8_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(int8_t, std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int8_t)
END_DISPATCH_TABLE
