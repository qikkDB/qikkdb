#include "GpuSqlDispatcherArithmeticUnaryFunctions.h"
#include <array>
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::absFunctions = {&GpuSqlDispatcher::arithmeticUnaryConst<ArithmeticUnaryOperations::_abs, int32_t>, &GpuSqlDispatcher::arithmeticUnaryCol<ArithmeticUnaryOperations::_abs, int32_t> };