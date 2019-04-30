#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../Context.h"
#include "../../DataType.h"
#include "../GPUWhereFunctions.h"
#include "GPUFilter.cuh"
#include "GPULogic.cuh"
#include "GPUArithmetic.cuh"
#include "GPUDate.cuh"
#include "GPUWhereInterpreter.cuh"
#include "MaybeDeref.cuh"
#include "GpuMemory.cuh"

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
		GPUStack<2048> gpuStack;
		for (int32_t j = 0; j < opCodesCount; j++)
		{
            opCodes[i].fun_ptr(opCodes[i], i, gpuStack, symbols);
		}
        outMask[i] = gpuStack.pop<int8_t>(); 
    }
}

__device__ void invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

__device__ void invalidContainsArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

__device__ GpuVMFunction add_gpu_greater_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_less_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_greaterEqual_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_lessEqual_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_equal_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_notEqual_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logicalAnd_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logicalOr_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_mul_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_div_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_add_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_sub_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_mod_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_push_function(int32_t dataTypes);

__global__ void kernel_fill_gpu_dispatch_table(GpuVMFunction * gpuDispatchPtr, size_t arraySize)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < arraySize; i += stride)
	{
		int32_t operation = i / GPUWhereFunctions::FUNC_COUNT;
		int32_t dataTypes = i % (DataType::COLUMN_INT * DataType::COLUMN_INT);

		switch (operation)
		{
		case 0:
			gpuDispatchPtr[i] = add_gpu_greater_function(dataTypes);
			break;
		case 1:
			gpuDispatchPtr[i] = add_gpu_less_function(dataTypes);
			break;
		case 2:
			gpuDispatchPtr[i] = add_gpu_greaterEqual_function(dataTypes);
			break;
		case 3:
			gpuDispatchPtr[i] = add_gpu_lessEqual_function(dataTypes);
			break;
		case 4:
			gpuDispatchPtr[i] = add_gpu_equal_function(dataTypes);
			break;
		case 5:
			gpuDispatchPtr[i] = add_gpu_notEqual_function(dataTypes);
			break;
		case 6:
			gpuDispatchPtr[i] = add_gpu_logicalAnd_function(dataTypes);
			break;
		case 7:
			gpuDispatchPtr[i] = add_gpu_logicalOr_function(dataTypes);
			break;
		case 8:
			gpuDispatchPtr[i] = add_gpu_mul_function(dataTypes);
			break;
		case 9:
			gpuDispatchPtr[i] = add_gpu_div_function(dataTypes);
			break;
		case 10:
			gpuDispatchPtr[i] = add_gpu_add_function(dataTypes);
			break;
		case 11:
			gpuDispatchPtr[i] = add_gpu_sub_function(dataTypes);
			break;
		case 12:
			gpuDispatchPtr[i] = add_gpu_mod_function(dataTypes);
			break;
		case 13:
			gpuDispatchPtr[i] = add_gpu_push_function(i % DataType::DATA_TYPE_SIZE);
		default:
			break;
		}
	}
}

__device__ GpuVMFunction add_gpu_greater_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::greater, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::greater, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::greater, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::greater, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::greater, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::greater, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::greater, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::greater, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::greater, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::greater, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::greater, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::greater, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::greater, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::greater, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::greater, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::greater, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::greater, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::greater, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::greater, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::greater, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::greater>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_less_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::less, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::less, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::less, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::less, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::less, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::less, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::less, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::less, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::less, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::less, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::less, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::less, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::less, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::less, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::less, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::less, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::less, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::less, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::less, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::less, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::less, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::less, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::less, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::less, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::less, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::less>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_greaterEqual_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::greaterEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::greaterEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::greaterEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::greaterEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::greaterEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::greaterEqual, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::greaterEqual, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::greaterEqual, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::greaterEqual, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::greaterEqual>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_lessEqual_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::lessEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::lessEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::lessEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::lessEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::lessEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::lessEqual, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::lessEqual, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::lessEqual, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::lessEqual, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::lessEqual, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::lessEqual>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_equal_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::equal, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::equal, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::equal, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::equal, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::equal, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::equal, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::equal, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::equal, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::equal, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::equal, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::equal, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::equal, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::equal, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::equal, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::equal, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::equal, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::equal, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::equal, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::equal, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::equal, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::equal>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_notEqual_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<FilterConditions::notEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<FilterConditions::notEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<FilterConditions::notEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunction<FilterConditions::notEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunction<FilterConditions::notEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::notEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::notEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<FilterConditions::notEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunction<FilterConditions::notEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunction<FilterConditions::notEqual, float, float>;
		break;
	case 35:
		return &filterFunction<FilterConditions::notEqual, float, double>;
		break;
	case 39:
		return &filterFunction<FilterConditions::notEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunction<FilterConditions::notEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunction<FilterConditions::notEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunction<FilterConditions::notEqual, double, float>;
		break;
	case 51:
		return &filterFunction<FilterConditions::notEqual, double, double>;
		break;
	case 55:
		return &filterFunction<FilterConditions::notEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunction<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<FilterConditions::notEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunction<FilterConditions::notEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunction<FilterConditions::notEqual, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::notEqual>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_logicalAnd_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<LogicOperations::logicalAnd, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<LogicOperations::logicalAnd, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<LogicOperations::logicalAnd, int32_t, float>;
		break;
	case 3:
		return &filterFunction<LogicOperations::logicalAnd, int32_t, double>;
		break;
	case 7:
		return &filterFunction<LogicOperations::logicalAnd, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, float>;
		break;
	case 19:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, double>;
		break;
	case 23:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<LogicOperations::logicalAnd, float, int32_t>;
		break;
	case 33:
		return &filterFunction<LogicOperations::logicalAnd, float, int64_t>;
		break;
	case 34:
		return &filterFunction<LogicOperations::logicalAnd, float, float>;
		break;
	case 35:
		return &filterFunction<LogicOperations::logicalAnd, float, double>;
		break;
	case 39:
		return &filterFunction<LogicOperations::logicalAnd, float, int8_t>;
		break;
	case 48:
		return &filterFunction<LogicOperations::logicalAnd, double, int32_t>;
		break;
	case 49:
		return &filterFunction<LogicOperations::logicalAnd, double, int64_t>;
		break;
	case 50:
		return &filterFunction<LogicOperations::logicalAnd, double, float>;
		break;
	case 51:
		return &filterFunction<LogicOperations::logicalAnd, double, double>;
		break;
	case 55:
		return &filterFunction<LogicOperations::logicalAnd, double, int8_t>;
		break;
	case 112:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, float>;
		break;
	case 115:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, double>;
		break;
	case 119:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<LogicOperations::logicalAnd>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_logicalOr_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunction<LogicOperations::logicalOr, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunction<LogicOperations::logicalOr, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunction<LogicOperations::logicalOr, int32_t, float>;
		break;
	case 3:
		return &filterFunction<LogicOperations::logicalOr, int32_t, double>;
		break;
	case 7:
		return &filterFunction<LogicOperations::logicalOr, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunction<LogicOperations::logicalOr, int64_t, float>;
		break;
	case 19:
		return &filterFunction<LogicOperations::logicalOr, int64_t, double>;
		break;
	case 23:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunction<LogicOperations::logicalOr, float, int32_t>;
		break;
	case 33:
		return &filterFunction<LogicOperations::logicalOr, float, int64_t>;
		break;
	case 34:
		return &filterFunction<LogicOperations::logicalOr, float, float>;
		break;
	case 35:
		return &filterFunction<LogicOperations::logicalOr, float, double>;
		break;
	case 39:
		return &filterFunction<LogicOperations::logicalOr, float, int8_t>;
		break;
	case 48:
		return &filterFunction<LogicOperations::logicalOr, double, int32_t>;
		break;
	case 49:
		return &filterFunction<LogicOperations::logicalOr, double, int64_t>;
		break;
	case 50:
		return &filterFunction<LogicOperations::logicalOr, double, float>;
		break;
	case 51:
		return &filterFunction<LogicOperations::logicalOr, double, double>;
		break;
	case 55:
		return &filterFunction<LogicOperations::logicalOr, double, int8_t>;
		break;
	case 112:
		return &filterFunction<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunction<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunction<LogicOperations::logicalOr, int8_t, float>;
		break;
	case 115:
		return &filterFunction<LogicOperations::logicalOr, int8_t, double>;
		break;
	case 119:
		return &filterFunction<LogicOperations::logicalOr, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<LogicOperations::logicalOr>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_mul_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int8_t>;
		break;
	case 32:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int32_t>;
		break;
	case 33:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int64_t>;
		break;
	case 34:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, float>;
		break;
	case 35:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, double>;
		break;
	case 39:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int8_t>;
		break;
	case 48:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int32_t>;
		break;
	case 49:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int64_t>;
		break;
	case 50:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, float>;
		break;
	case 51:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, double>;
		break;
	case 55:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int8_t>;
		break;
	case 112:
		return &arithmeticFunction<ArithmeticOperations::mul, int8_t, int8_t, int32_t>;
		break;
	case 113:
		return &arithmeticFunction<ArithmeticOperations::mul, int8_t, int8_t, int64_t>;
		break;
	case 114:
		return &arithmeticFunction<ArithmeticOperations::mul, int8_t, int8_t, float>;
		break;
	case 115:
		return &arithmeticFunction<ArithmeticOperations::mul, int8_t, int8_t, double>;
		break;
	case 119:
		return &arithmeticFunction<ArithmeticOperations::mul, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::mul>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_div_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int8_t>;
		break;
	case 32:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int32_t>;
		break;
	case 33:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int64_t>;
		break;
	case 34:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, float>;
		break;
	case 35:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, double>;
		break;
	case 39:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int8_t>;
		break;
	case 48:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int32_t>;
		break;
	case 49:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int64_t>;
		break;
	case 50:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, float>;
		break;
	case 51:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, double>;
		break;
	case 55:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int8_t>;
		break;
	case 112:
		return &arithmeticFunction<ArithmeticOperations::div, int8_t, int8_t, int32_t>;
		break;
	case 113:
		return &arithmeticFunction<ArithmeticOperations::div, int8_t, int8_t, int64_t>;
		break;
	case 114:
		return &arithmeticFunction<ArithmeticOperations::div, int8_t, int8_t, float>;
		break;
	case 115:
		return &arithmeticFunction<ArithmeticOperations::div, int8_t, int8_t, double>;
		break;
	case 119:
		return &arithmeticFunction<ArithmeticOperations::div, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::div>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_add_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int8_t>;
		break;
	case 32:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int32_t>;
		break;
	case 33:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int64_t>;
		break;
	case 34:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, float>;
		break;
	case 35:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, double>;
		break;
	case 39:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int8_t>;
		break;
	case 48:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int32_t>;
		break;
	case 49:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int64_t>;
		break;
	case 50:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, float>;
		break;
	case 51:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, double>;
		break;
	case 55:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int8_t>;
		break;
	case 112:
		return &arithmeticFunction<ArithmeticOperations::add, int8_t, int8_t, int32_t>;
		break;
	case 113:
		return &arithmeticFunction<ArithmeticOperations::add, int8_t, int8_t, int64_t>;
		break;
	case 114:
		return &arithmeticFunction<ArithmeticOperations::add, int8_t, int8_t, float>;
		break;
	case 115:
		return &arithmeticFunction<ArithmeticOperations::add, int8_t, int8_t, double>;
		break;
	case 119:
		return &arithmeticFunction<ArithmeticOperations::add, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::add>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_sub_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int8_t>;
		break;
	case 32:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int32_t>;
		break;
	case 33:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int64_t>;
		break;
	case 34:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, float>;
		break;
	case 35:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, double>;
		break;
	case 39:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int8_t>;
		break;
	case 48:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int32_t>;
		break;
	case 49:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int64_t>;
		break;
	case 50:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, float>;
		break;
	case 51:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, double>;
		break;
	case 55:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int8_t>;
		break;
	case 112:
		return &arithmeticFunction<ArithmeticOperations::sub, int8_t, int8_t, int32_t>;
		break;
	case 113:
		return &arithmeticFunction<ArithmeticOperations::sub, int8_t, int8_t, int64_t>;
		break;
	case 114:
		return &arithmeticFunction<ArithmeticOperations::sub, int8_t, int8_t, float>;
		break;
	case 115:
		return &arithmeticFunction<ArithmeticOperations::sub, int8_t, int8_t, double>;
		break;
	case 119:
		return &arithmeticFunction<ArithmeticOperations::sub, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::sub>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_mod_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::mod, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::mod, int32_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::mod, int32_t, int32_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int64_t>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int8_t>;
		break;
	case 112:
		return &arithmeticFunction<ArithmeticOperations::mod, int8_t, int8_t, int32_t>;
		break;
	case 113:
		return &arithmeticFunction<ArithmeticOperations::mod, int8_t, int8_t, int64_t>;
		break;
	case 119:
		return &arithmeticFunction<ArithmeticOperations::mod, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::mod>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_push_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case CONST_INT:
		return &pushConstFunction<int32_t>;
	case CONST_LONG:
		return &pushConstFunction<int64_t>;
	case CONST_FLOAT:
		return &pushConstFunction<float>;
	case CONST_DOUBLE:
		return &pushConstFunction<double>;
	case CONST_POINT:
		return &invalidArgumentTypeHandler;
	case CONST_POLYGON:
		return &invalidArgumentTypeHandler;
	case CONST_STRING:
		return &invalidArgumentTypeHandler;
	case CONST_INT8_T:
		return &pushConstFunction<int8_t>;
	case COLUMN_INT:
		return &pushColFunction<int32_t>;
	case COLUMN_LONG:
		return &pushColFunction<int64_t>;
	case COLUMN_FLOAT:
		return &pushColFunction<float>;
	case COLUMN_DOUBLE:
		return &pushColFunction<float>;
	case COLUMN_POINT:
		return &invalidArgumentTypeHandler;
	case COLUMN_POLYGON:
		return &invalidArgumentTypeHandler;
	case COLUMN_STRING:
		return &invalidArgumentTypeHandler;
	case COLUMN_INT8_T:
		return &pushColFunction<int8_t>;
	default:
		return &invalidArgumentTypeHandler;
		break;
	}
}