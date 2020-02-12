#include "GpuSqlDispatcherGeoFunctions.h"
#include <array>
#include "DispatcherMacros.h"

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