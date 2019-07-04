#include "GpuSqlJoinFunctions.h"

std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinGreaterFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::greater, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greater, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greater, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greater, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greater, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greater, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greater, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greater, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greater, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greater, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greater, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greater, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greater, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greater, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greater, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greater, int8_t> };
std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinLessFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::less, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::less, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::less, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::less, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::less, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::less, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::less, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::less, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::less, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::less, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::less, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::less, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::less, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::less, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::less, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::less, int8_t> };
std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinGreaterEqualFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::greaterEqual, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greaterEqual, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greaterEqual, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greaterEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greaterEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greaterEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::greaterEqual, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::greaterEqual, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greaterEqual, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greaterEqual, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greaterEqual, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greaterEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greaterEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greaterEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::greaterEqual, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::greaterEqual, int8_t> };
std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinLessEqualFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::lessEqual, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::lessEqual, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::lessEqual, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::lessEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::lessEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::lessEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::lessEqual, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::lessEqual, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::lessEqual, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::lessEqual, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::lessEqual, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::lessEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::lessEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::lessEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::lessEqual, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::lessEqual, int8_t> };
std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinEqualFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::equal, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::equal, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::equal, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::equal, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::equal, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::equal, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::equal, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::equal, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::equal, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::equal, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::equal, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::equal, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::equal, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::equal, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::equal, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::equal, int8_t> };
std::array<GpuSqlJoinDispatcher::DispatchJoinFunction, DataType::DATA_TYPE_SIZE> GpuSqlJoinDispatcher::joinNotEqualFunctions = { &GpuSqlJoinDispatcher::joinConst<FilterConditions::notEqual, int32_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::notEqual, int64_t>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::notEqual, float>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::notEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::notEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::notEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerConst<FilterConditions::notEqual, std::string>, &GpuSqlJoinDispatcher::joinConst<FilterConditions::notEqual, int8_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::notEqual, int32_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::notEqual, int64_t>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::notEqual, float>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::notEqual, double>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::notEqual, ColmnarDB::Types::Point>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::notEqual, ColmnarDB::Types::ComplexPolygon>, &GpuSqlJoinDispatcher::invalidOperandTypesErrorHandlerCol<FilterConditions::notEqual, std::string>, &GpuSqlJoinDispatcher::joinCol<FilterConditions::notEqual, int8_t> };
GpuSqlJoinDispatcher::DispatchJoinFunction GpuSqlJoinDispatcher::joinDoneFunction = &GpuSqlJoinDispatcher::joinDone;