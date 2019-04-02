#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../Context.h"
#include "../../DataType.h"
#include "GPUFilter.cuh"
#include "GPULogic.cuh"
#include "GPUArithmetic.cuh"
#include "GPUDate.cuh"
#include "GPUDispatch.cuh"
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

__device__ void containsFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	GPUMemory::GPUPolygon p;
	p.pointCount = gpuStack.pop<int32_t*>();
	p.pointIdx = gpuStack.pop<int32_t*>();
	p.polyCount = gpuStack.pop<int32_t*>();
	p.polyIdx = gpuStack.pop<int32_t*>();
	p.polyPoints = gpuStack.pop<NativeGeoPoint*>();
	//Zavolaj Contains,urob z neho device funkiu
}


__device__ void invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

__device__ void invalidContainsArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

__device__ DispatchFunction add_gpu_greater_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_less_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_greaterEqual_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_lessEqual_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_equal_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_notEqual_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_logicalAnd_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_logicalOr_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_mul_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_div_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_add_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_sub_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_mod_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_contains_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_logicalNot_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_year_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_month_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_day_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_hour_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_minute_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_second_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_pushCol_function(int32_t dataTypes);
__device__ DispatchFunction add_gpu_pushConst_function(int32_t dataTypes);

__global__ void kernel_fill_gpu_dispatch_table(DispatchFunction * gpuDispatchPtr, size_t arraySize)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < arraySize; i += stride)
	{
		int32_t operation = i / OPERATIONS_COUNT;
		int32_t dataTypes = i % (DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE);

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
			gpuDispatchPtr[i] = add_gpu_contains_function(dataTypes);
			break;
		case 14:
			gpuDispatchPtr[i] = add_gpu_logicalNot_function(dataTypes);
			break;
		case 15:
			gpuDispatchPtr[i] = add_gpu_year_function(dataTypes);
			break;
		case 16:
			gpuDispatchPtr[i] = add_gpu_month_function(dataTypes);
			break;
		case 17:
			gpuDispatchPtr[i] = add_gpu_day_function(dataTypes);
			break;
		case 18:
			gpuDispatchPtr[i] = add_gpu_hour_function(dataTypes);
			break;
		case 19:
			gpuDispatchPtr[i] = add_gpu_minute_function(dataTypes);
			break;
		case 20:
			gpuDispatchPtr[i] = add_gpu_second_function(dataTypes);
			break;
		case 21:
			gpuDispatchPtr[i] = add_gpu_pushCol_function(dataTypes);
			break;
		case 22:
			gpuDispatchPtr[i] = add_gpu_pushConst_function(dataTypes);
			break;
		default:
			break;
		}
	}
}

__device__ DispatchFunction add_gpu_greater_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_less_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_greaterEqual_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_lessEqual_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_equal_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_notEqual_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_logicalAnd_function(int32_t dataTypes)
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

__device__ DispatchFunction add_gpu_logicalOr_function(int32_t dataTypes)
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

__device__ DispatchFunction add_gpu_mul_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_div_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_add_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_sub_function(int32_t dataTypes)
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


__device__ DispatchFunction add_gpu_mod_function(int32_t dataTypes)
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

__device__ DispatchFunction add_gpu_logicalNot_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 1:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 2:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 3:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 7:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 16:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 17:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 18:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 19:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 23:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 32:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 33:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 34:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 35:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 39:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 48:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 49:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 50:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 51:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 55:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 112:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	case 113:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	case 114:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	case 115:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	case 119:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<LogicOperations::logicalNot>;
		break;
	}
}

__device__ DispatchFunction add_gpu_contains_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 84:
		return &containsFunction;
		break;
	default:
		return &invalidContainsArgumentTypeHandler;
		break;
	}
}

__device__ DispatchFunction add_gpu_year_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::year>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::year>;
		break;
	}
}


__device__ DispatchFunction add_gpu_month_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::month>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::month>;
		break;
	}
}


__device__ DispatchFunction add_gpu_day_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::day>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::day>;
		break;
	}
}


__device__ DispatchFunction add_gpu_hour_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::hour>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::hour>;
		break;
	}
}


__device__ DispatchFunction add_gpu_minute_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::minute>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::minute>;
		break;
	}
}


__device__ DispatchFunction add_gpu_second_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 17:
		return &dateFunction<DateOperations::second>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::second>;
		break;
	}
}

__device__ DispatchFunction add_gpu_pushCol_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &pushColFunction<int32_t>;
		break;
	case 1:
		return &pushColFunction<int32_t>;
		break;
	case 2:
		return &pushColFunction<int32_t>;
		break;
	case 3:
		return &pushColFunction<int32_t>;
		break;
	case 7:
		return &pushColFunction<int32_t>;
		break;
	case 16:
		return &pushColFunction<int64_t>;
		break;
	case 17:
		return &pushColFunction<int64_t>;
		break;
	case 18:
		return &pushColFunction<int64_t>;
		break;
	case 19:
		return &pushColFunction<int64_t>;
		break;
	case 23:
		return &pushColFunction<int64_t>;
		break;
	case 32:
		return &pushColFunction<float>;
		break;
	case 33:
		return &pushColFunction<float>;
		break;
	case 34:
		return &pushColFunction<float>;
		break;
	case 35:
		return &pushColFunction<float>;
		break;
	case 39:
		return &pushColFunction<float>;
		break;
	case 48:
		return &pushColFunction<double>;
		break;
	case 49:
		return &pushColFunction<double>;
		break;
	case 50:
		return &pushColFunction<double>;
		break;
	case 51:
		return &pushColFunction<double>;
		break;
	case 55:
		return &pushColFunction<double>;
		break;
	case 112:
		return &pushColFunction<int8_t>;
		break;
	case 113:
		return &pushColFunction<int8_t>;
		break;
	case 114:
		return &pushColFunction<int8_t>;
		break;
	case 115:
		return &pushColFunction<int8_t>;
		break;
	case 119:
		return &pushColFunction<int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler;
		break;
	}
}


__device__ DispatchFunction add_gpu_pushConst_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &pushConstFunction<int32_t>;
		break;
	case 1:
		return &pushConstFunction<int32_t>;
		break;
	case 2:
		return &pushConstFunction<int32_t>;
		break;
	case 3:
		return &pushConstFunction<int32_t>;
		break;
	case 7:
		return &pushConstFunction<int32_t>;
		break;
	case 16:
		return &pushConstFunction<int64_t>;
		break;
	case 17:
		return &pushConstFunction<int64_t>;
		break;
	case 18:
		return &pushConstFunction<int64_t>;
		break;
	case 19:
		return &pushConstFunction<int64_t>;
		break;
	case 23:
		return &pushConstFunction<int64_t>;
		break;
	case 32:
		return &pushConstFunction<float>;
		break;
	case 33:
		return &pushConstFunction<float>;
		break;
	case 34:
		return &pushConstFunction<float>;
		break;
	case 35:
		return &pushConstFunction<float>;
		break;
	case 39:
		return &pushConstFunction<float>;
		break;
	case 48:
		return &pushConstFunction<double>;
		break;
	case 49:
		return &pushConstFunction<double>;
		break;
	case 50:
		return &pushConstFunction<double>;
		break;
	case 51:
		return &pushConstFunction<double>;
		break;
	case 55:
		return &pushConstFunction<double>;
		break;
	case 112:
		return &pushConstFunction<int8_t>;
		break;
	case 113:
		return &pushConstFunction<int8_t>;
		break;
	case 114:
		return &pushConstFunction<int8_t>;
		break;
	case 115:
		return &pushConstFunction<int8_t>;
		break;
	case 119:
		return &pushConstFunction<int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler;
		break;
	}
}