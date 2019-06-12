#include "GpuSqlDispatcherStringFunctions.h"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include <array>

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::ltrimFunctions = { &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryConst<StringUnaryOperations::ltrim, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::ltrim, int8_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryCol<StringUnaryOperations::ltrim, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::ltrim, int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::rtrimFunctions = { &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryConst<StringUnaryOperations::rtrim, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::rtrim, int8_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryCol<StringUnaryOperations::rtrim, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::rtrim, int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::lowerFunctions = { &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryConst<StringUnaryOperations::lower, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::lower, int8_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryCol<StringUnaryOperations::lower, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::lower, int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::upperFunctions = { &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryConst<StringUnaryOperations::upper, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<StringUnaryOperations::upper, int8_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, int32_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, int64_t>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, float>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::stringUnaryCol<StringUnaryOperations::upper, std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<StringUnaryOperations::upper, int8_t> };
