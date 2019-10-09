#include "GpuSqlDispatcherArithmeticUnaryFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"
#include "DispatcherMacros.h"

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::minusFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::absoluteFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::sineFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::cosineFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::tangentFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::cotangentFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::arcsineFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::arccosineFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::arctangentFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::logarithm10Functions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::logarithmNaturalFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::exponentialFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::squareRootFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::squareFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::signFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::roundFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::floorFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, int8_t)
END_DISPATCHER_TABLE

BEGIN_DISPATCHER_UNARY_TABLE(GpuSqlDispatcher::ceilFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, int8_t)
END_DISPATCHER_TABLE
