#include "GpuSqlDispatcherBinaryFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUBinary.cuh"
#define MERGED
#include "DispatcherMacros.h"

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::mulFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mul, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mul, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mul, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mul, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::divFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::div, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::div, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::div, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::div, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::addFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::add, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::add, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::add, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::add, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::subFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::sub, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::sub, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::sub, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::sub, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::modFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mod, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::mod, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseOrFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseOr, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseOr, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseAndFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseAnd, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseAnd, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseXorFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseXor, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseXor, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseLeftShiftFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseLeftShift, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseLeftShift, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseRightShiftFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseRightShift, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::bitwiseRightShift, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::powerFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::power, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::power, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::power, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::power, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logarithmFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::logarithm, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::logarithm, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::logarithm, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::logarithm, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::arctangent2Functions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::arctangent2, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::arctangent2, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::arctangent2, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::arctangent2, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::roundDecimalFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::roundDecimal, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::roundDecimal, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::roundDecimal, float, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::roundDecimal, double, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::rootFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::root, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::root, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::root, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::root, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoLongitudeToTileXFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLongitudeToTileX, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLongitudeToTileX, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLongitudeToTileX, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLongitudeToTileX, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoLatitudeToTileYFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLatitudeToTileY, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLatitudeToTileY, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLatitudeToTileY, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoLatitudeToTileY, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoTileXToLongitudeFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileXToLongitude, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileXToLongitude, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileXToLongitude, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileXToLongitude, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoTileYToLatitudeFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileYToLatitude, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileYToLatitude, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileYToLatitude, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ArithmeticOperations::geoTileYToLatitude, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::leftFunctions_)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, StringBinaryOperations::left, std::string, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::left, QikkDB::Types::ComplexPolygon)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::rightFunctions_)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, StringBinaryOperations::right, std::string, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::right, QikkDB::Types::ComplexPolygon)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::concatFunctions)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, int32_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, int64_t)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, float)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, double)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, StringBinaryOperations::concat, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_INVALID_TYPE(StringBinaryOperations::concat, QikkDB::Types::ComplexPolygon)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::pointFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ConversionOperations::latLonToPoint, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ConversionOperations::latLonToPoint, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ConversionOperations::latLonToPoint, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, ConversionOperations::latLonToPoint, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ConversionOperations::latLonToPoint, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(ConversionOperations::latLonToPoint, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ConversionOperations::latLonToPoint, std::string)
DISPATCHER_INVALID_TYPE(ConversionOperations::latLonToPoint, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::intersectFunctions_)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int32_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int64_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, float)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, double)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, QikkDB::Types::Point)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, PolygonFunctions::polyIntersect, QikkDB::Types::ComplexPolygon, 0, 0, 0, 0, 0, 1, 0, 0)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, std::string)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyIntersect, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::unionFunctions_)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int32_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int64_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, float)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, double)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, QikkDB::Types::Point)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, PolygonFunctions::polyUnion, QikkDB::Types::ComplexPolygon, 0, 0, 0, 0, 0, 1, 0, 0)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, std::string)
DISPATCHER_INVALID_TYPE(PolygonFunctions::polyUnion, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::containsFunctions_)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, int32_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, int64_t)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, float)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, double)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, QikkDB::Types::Point)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, PolygonFunctions::contains, QikkDB::Types::ComplexPolygon, 0, 0, 0, 0, 1, 0, 0, 0)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, std::string)
DISPATCHER_INVALID_TYPE(PolygonFunctions::contains, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::greaterFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::greater, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::greater, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greater, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::lessFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::less, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::less, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::less, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::greaterEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::greaterEqual, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::greaterEqual, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::greaterEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::lessEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::lessEqual, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::lessEqual, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::lessEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::equalFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::equal, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::equal, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::equal, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::notEqualFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::notEqual, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::notEqual, QikkDB::Types::ComplexPolygon)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, std::string, 0, 0, 0, 0, 0, 0, 1, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::notEqual, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logicalAndFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalAnd, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalAnd, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalAnd, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalAnd, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalAnd, std::string)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalAnd, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logicalOrFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalOr, int32_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalOr, int64_t, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalOr, float, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalOr, double, 1, 1, 1, 1, 0, 0, 0, 1)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, QikkDB::Types::Point)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, QikkDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(FilterConditions::logicalOr, std::string)
DISPATCHER_TYPE(GpuSqlDispatcher::Binary, FilterConditions::logicalOr, int8_t, 1, 1, 1, 1, 0, 0, 0, 1)
END_DISPATCH_TABLE


#undef MERGED
