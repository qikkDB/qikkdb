#include "GpuSqlDispatcherLogicFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#define MERGED
#include "DispatcherMacros.h"

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