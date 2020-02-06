#include "GpuSqlDispatcherArithmeticFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "DispatcherMacros.h"

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::mulFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mul, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mul, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mul, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mul, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mul, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::divFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::div, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::div, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::div, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::div, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::div, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::addFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::add, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::add, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::add, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::add, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::add, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::subFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::sub, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::sub, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::sub, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::sub, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::sub, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::modFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mod, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::mod, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::mod, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseOrFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseOr, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseOr, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseOr, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseAndFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseAnd, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseAnd, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseAnd, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseXorFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseXor, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseXor, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseXor, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseLeftShiftFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseLeftShift, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseLeftShift, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseLeftShift, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::bitwiseRightShiftFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseRightShift, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::bitwiseRightShift, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, float)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, double)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::bitwiseRightShift, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::powerFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::power, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::power, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::power, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::power, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::power, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::logarithmFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::logarithm, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::logarithm, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::logarithm, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::logarithm, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::logarithm, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::arctangent2Functions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::arctangent2, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::arctangent2, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::arctangent2, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::arctangent2, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::arctangent2, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::roundDecimalFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::roundDecimal, int32_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::roundDecimal, int64_t, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::roundDecimal, float, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::roundDecimal, double, 1, 1, 0, 0, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::roundDecimal, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::rootFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::root, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::root, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::root, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::root, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::root, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoLongitudeToTileXFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLongitudeToTileX, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLongitudeToTileX, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLongitudeToTileX, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLongitudeToTileX, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLongitudeToTileX, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoLatitudeToTileYFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLatitudeToTileY, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLatitudeToTileY, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLatitudeToTileY, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoLatitudeToTileY, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoLatitudeToTileY, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoTileXToLongitudeFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileXToLongitude, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileXToLongitude, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileXToLongitude, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileXToLongitude, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileXToLongitude, int8_t)
END_DISPATCH_TABLE

BEGIN_DISPATCH_TABLE(GpuSqlDispatcher::geoTileYToLatitudeFunctions_)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileYToLatitude, int32_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileYToLatitude, int64_t, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileYToLatitude, float, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_TYPE(GpuSqlDispatcher::Arithmetic, ArithmeticOperations::geoTileYToLatitude, double, 1, 1, 1, 1, 0, 0, 0, 0)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, ColmnarDB::Types::Point)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, std::string)
DISPATCHER_INVALID_TYPE(ArithmeticOperations::geoTileYToLatitude, int8_t)
END_DISPATCH_TABLE
