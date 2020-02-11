#include "GpuSqlDispatcherStringFunctions.h"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"
#include <array>
#include "DispatcherMacros.h"

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::concatFunctions)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, int32_t)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, int64_t)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, float)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, double)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, ColmnarDB::Types::Point)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_FUNCTION(GpuSqlDispatcher::StringBinary, StringBinaryOperations::concat)
DISPATCHER_ERROR(StringBinaryOperations::concat, std::string, int8_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, ColmnarDB::Types::ComplexPolygon)
END_DISPATCH_TABLE
