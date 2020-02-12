#include "GpuSqlDispatcherGeoFunctions.h"
#include <array>
#include "DispatcherMacros.h"

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::intersectFunctions_)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int32_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int64_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, float)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, double)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, ColmnarDB::Types::Point)
DISPATCHER_TYPE(GpuSqlDispatcher::PolygonOperation,
                PolygonFunctions::polyIntersect,
                ColmnarDB::Types::ComplexPolygon,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, std::string)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::unionFunctions_)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int32_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int64_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, float)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, double)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, ColmnarDB::Types::Point)
DISPATCHER_TYPE(GpuSqlDispatcher::PolygonOperation,
                PolygonFunctions::polyUnion,
                ColmnarDB::Types::ComplexPolygon,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, std::string)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::containsFunctions_)
DISPATCHER_INVALID_TYPE_NOOP(int32_t)
DISPATCHER_INVALID_TYPE_NOOP(int64_t)
DISPATCHER_INVALID_TYPE_NOOP(float)
DISPATCHER_INVALID_TYPE_NOOP(double)
DISPATCHER_INVALID_TYPE_NOOP(ColmnarDB::Types::Point)
DISPATCHER_TYPE_NOOP(GpuSqlDispatcher::Contains, ColmnarDB::Types::ComplexPolygon, 0, 0, 0, 0, 1, 0, 0, 0)
DISPATCHER_INVALID_TYPE_NOOP(std::string)
DISPATCHER_INVALID_TYPE_NOOP(int8_t)
END_DISPATCH_TABLE