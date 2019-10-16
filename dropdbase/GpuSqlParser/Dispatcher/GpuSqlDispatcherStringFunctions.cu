#include "GpuSqlDispatcherStringFunctions.h"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"
#include <array>
#include "DispatcherMacros.h"
BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::ltrimFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnary, StringUnaryOperations::ltrim)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::rtrimFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnary, StringUnaryOperations::rtrim)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::lowerFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnary, StringUnaryOperations::lower)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::upperFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnary, StringUnaryOperations::upper)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::reverseFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnary, StringUnaryOperations::reverse)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::lenFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, float)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, double)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::StringUnaryNumeric, StringUnaryNumericOperations::len)
DISPATCHER_UNARY_ERROR(StringUnaryNumericOperations::len, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::leftFunctions_)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE_SPECIFIC(GpuSqlDispatcher::StringBinaryNumeric, StringBinaryOperations::left, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, ColmnarDB::Types::ComplexPolygon)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::rightFunctions_)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_TYPE_SPECIFIC(GpuSqlDispatcher::StringBinaryNumeric, StringBinaryOperations::right, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, ColmnarDB::Types::ComplexPolygon)
END_DISPATCH_TABLE

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
