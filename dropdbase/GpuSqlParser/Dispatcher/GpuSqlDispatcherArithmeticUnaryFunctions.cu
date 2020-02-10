#include "GpuSqlDispatcherArithmeticUnaryFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#define MERGED
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::minusFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::minus, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::minus, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::absoluteFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::absolute, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::absolute, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::sineFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sine, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::cosineFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cosine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cosine, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::tangentFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::tangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::tangent, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::cotangentFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::cotangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::cotangent, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::arcsineFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arcsine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arcsine, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::arccosineFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arccosine, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arccosine, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::arctangentFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::arctangent, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::arctangent, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::logarithm10Functions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithm10, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithm10, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::logarithmNaturalFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::logarithmNatural, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::logarithmNatural, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::exponentialFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::exponential, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::exponential, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::squareRootFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::squareRoot, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::squareRoot, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::squareFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::square, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::square, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::signFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::sign, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::sign, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::roundFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::round, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::round, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::floorFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::floor, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::floor, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::ceilFunctions_)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, int64_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, float)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, ArithmeticUnaryOperations::ceil, double)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, std::string)
DISPATCHER_UNARY_ERROR(ArithmeticUnaryOperations::ceil, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::yearFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::year, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::year, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::year, float)
DISPATCHER_UNARY_ERROR(DateOperations::year, double)
DISPATCHER_UNARY_ERROR(DateOperations::year, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::year, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::year, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::year, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::monthFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::month, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::month, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::month, float)
DISPATCHER_UNARY_ERROR(DateOperations::month, double)
DISPATCHER_UNARY_ERROR(DateOperations::month, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::month, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::month, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::month, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::dayFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::day, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::day, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::day, float)
DISPATCHER_UNARY_ERROR(DateOperations::day, double)
DISPATCHER_UNARY_ERROR(DateOperations::day, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::day, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::day, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::day, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::hourFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::hour, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::hour, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::hour, float)
DISPATCHER_UNARY_ERROR(DateOperations::hour, double)
DISPATCHER_UNARY_ERROR(DateOperations::hour, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::hour, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::hour, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::hour, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::minuteFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::minute, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::minute, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::minute, float)
DISPATCHER_UNARY_ERROR(DateOperations::minute, double)
DISPATCHER_UNARY_ERROR(DateOperations::minute, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::minute, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::minute, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::minute, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::secondFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::second, int32_t)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, DateOperations::second, int64_t)
DISPATCHER_UNARY_ERROR(DateOperations::second, float)
DISPATCHER_UNARY_ERROR(DateOperations::second, double)
DISPATCHER_UNARY_ERROR(DateOperations::second, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::second, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::second, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::second, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::ltrimFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, StringUnaryOperations::ltrim, std::string)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::ltrim, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::rtrimFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, StringUnaryOperations::rtrim, std::string)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::rtrim, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::lowerFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, StringUnaryOperations::lower, std::string)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::lower, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::upperFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, StringUnaryOperations::upper, std::string)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::upper, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::reverseFunctions_)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int32_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int64_t)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, float)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, double)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION_MERGED(GpuSqlDispatcher::ArithmeticUnary, StringUnaryOperations::reverse, std::string)
DISPATCHER_UNARY_ERROR(StringUnaryOperations::reverse, int8_t)
END_DISPATCH_TABLE

#undef MERGED
