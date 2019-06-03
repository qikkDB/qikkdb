#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../Context.h"
#include "../../DataType.h"
#include "../DispatcherFunction.h"
#include "GPUFilter.cuh"
#include "GPULogic.cuh"
#include "GPUArithmetic.cuh"
#include "GPUArithmeticUnary.cuh"
#include "GPUDate.cuh"
#include "GPUWhereInterpreter.cuh"
#include "MaybeDeref.cuh"
#include "GPUMemory.cuh"
#include "GPUPolygonContains.cuh"

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
		GPUStack<2048> gpuStack;
		for (int32_t j = 0; j < opCodesCount; j++)
		{
            opCodes[j].fun_ptr(opCodes[j], i, gpuStack, symbols);
			__syncthreads();
		}
        outMask[i] = gpuStack.pop<int8_t>(); 
    }
}

__device__ void containsColPolyFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	GPUMemory::GPUPolygon p;
	p.pointCount = gpuStack.pop<int32_t*>();
	p.pointIdx = gpuStack.pop<int32_t*>();
	p.polyCount = gpuStack.pop<int32_t*>();
	p.polyIdx = gpuStack.pop<int32_t*>();
	p.polyPoints = gpuStack.pop<NativeGeoPoint*>();
	NativeGeoPoint point = gpuStack.pop<NativeGeoPoint>();
	gpuStack.push<int8_t>(point_in_polygon(offset,p,point));
}

__device__ void containsValsFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	GPUMemory::GPUPolygon p;
	p.pointCount = gpuStack.pop<int32_t*>();
	p.pointIdx = gpuStack.pop<int32_t*>();
	p.polyCount = gpuStack.pop<int32_t*>();
	p.polyIdx = gpuStack.pop<int32_t*>();
	p.polyPoints = gpuStack.pop<NativeGeoPoint*>();
	NativeGeoPoint point = gpuStack.pop<NativeGeoPoint>();
	gpuStack.push<int8_t>(point_in_polygon(0,p,point));
}

__device__ void invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	gpuStack.push(0);
}

__device__ void invalidContainsArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

template <>
__device__ void pushConstFunction<NativeGeoPoint>(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	NativeGeoPoint gp;
	gp.latitude = (*reinterpret_cast<float*>(opCode.data));
	gp.longitude = (*reinterpret_cast<float*>(opCode.data) + sizeof(float));
	gpuStack.push<NativeGeoPoint>(gp);
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
__device__ GpuVMFunction add_gpu_bitwiseOr_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_bitwiseAnd_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_bitwiseXor_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_bitwiseLeftShift_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_bitwiseRightShift_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logarithm_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_power_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_root_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_arctangent2_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logicalNot_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_minus_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_year_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_month_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_day_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_hour_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_minute_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_second_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_absolute_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_sine_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_cosine_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_tangent_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_cotangent_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_arcsine_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_arccosine_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_arctangent_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logarithm10_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_logarithmNatural_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_exponential_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_squareRoot_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_square_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_sign_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_round_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_floor_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_ceil_function(int32_t dataTypes);
__device__ GpuVMFunction add_gpu_push_function(int32_t dataTypes);

__global__ void kernel_fill_gpu_dispatch_table(GpuVMFunction * gpuDispatchPtr, size_t arraySize)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < arraySize; i += stride)
	{
		DispatcherFunction operation = static_cast<DispatcherFunction>(i / (DataType::COLUMN_INT * DataType::COLUMN_INT));
		int32_t dataTypes = i % (DataType::COLUMN_INT * DataType::COLUMN_INT);
		switch (operation)
		{
		case DispatcherFunction::GT_FUNC:
			gpuDispatchPtr[i] = add_gpu_greater_function(dataTypes);
			break;
		case DispatcherFunction::LT_FUNC:
			gpuDispatchPtr[i] = add_gpu_less_function(dataTypes);
			break;
		case DispatcherFunction::GTEQ_FUNC:
			gpuDispatchPtr[i] = add_gpu_greaterEqual_function(dataTypes);
			break;
		case DispatcherFunction::LTEQ_FUNC:
			gpuDispatchPtr[i] = add_gpu_lessEqual_function(dataTypes);
			break;
		case DispatcherFunction::EQ_FUNC:
			gpuDispatchPtr[i] = add_gpu_equal_function(dataTypes);
			break;
		case DispatcherFunction::NEQ_FUNC:
			gpuDispatchPtr[i] = add_gpu_notEqual_function(dataTypes);
			break;
		case DispatcherFunction::AND_FUNC:
			gpuDispatchPtr[i] = add_gpu_logicalAnd_function(dataTypes);
			break;
		case DispatcherFunction::OR_FUNC:
			gpuDispatchPtr[i] = add_gpu_logicalOr_function(dataTypes);
			break;
		case DispatcherFunction::MUL_FUNC:
			gpuDispatchPtr[i] = add_gpu_mul_function(dataTypes);
			break;
		case DispatcherFunction::DIV_FUNC:
			gpuDispatchPtr[i] = add_gpu_div_function(dataTypes);
			break;
		case DispatcherFunction::ADD_FUNC:
			gpuDispatchPtr[i] = add_gpu_add_function(dataTypes);
			break;
		case DispatcherFunction::SUB_FUNC:
			gpuDispatchPtr[i] = add_gpu_sub_function(dataTypes);
			break;
		case DispatcherFunction::MOD_FUNC:
			gpuDispatchPtr[i] = add_gpu_mod_function(dataTypes);
			break;
		case DispatcherFunction::BIT_OR_FUNC:
			gpuDispatchPtr[i] = add_gpu_bitwiseOr_function(dataTypes);
			break;
		case DispatcherFunction::BIT_AND_FUNC:
			gpuDispatchPtr[i] = add_gpu_bitwiseAnd_function(dataTypes);
			break;
		case DispatcherFunction::BIT_XOR_FUNC:
			gpuDispatchPtr[i] = add_gpu_bitwiseXor_function(dataTypes);
			break;
		case DispatcherFunction::LEFT_SHIFT_FUNC:
			gpuDispatchPtr[i] = add_gpu_bitwiseLeftShift_function(dataTypes);
			break;
		case DispatcherFunction::RIGHT_SHIFT_FUNC:
			gpuDispatchPtr[i] = add_gpu_bitwiseRightShift_function(dataTypes);
			break;
		case DispatcherFunction::LOG_BIN_FUNC:
			gpuDispatchPtr[i] = add_gpu_logarithm_function(dataTypes);
			break;
		case DispatcherFunction::POW_BIN_FUNC:
			gpuDispatchPtr[i] = add_gpu_power_function(dataTypes);
			break;
		case DispatcherFunction::ROOT_BIN_FUNC:
			gpuDispatchPtr[i] = add_gpu_root_function(dataTypes);
			break;
		case DispatcherFunction::ATAN2_FUNC:
			gpuDispatchPtr[i] = add_gpu_arctangent2_function(dataTypes);
			break;
		case DispatcherFunction::NOT_FUNC:
			gpuDispatchPtr[i] = add_gpu_logicalNot_function(dataTypes);
			break;
		case DispatcherFunction::MINUS_FUNC:
			gpuDispatchPtr[i] = add_gpu_minus_function(dataTypes);
			break;
		case DispatcherFunction::YEAR_FUNC:
			gpuDispatchPtr[i] = add_gpu_year_function(dataTypes);
			break;
		case DispatcherFunction::MONTH_FUNC:
			gpuDispatchPtr[i] = add_gpu_month_function(dataTypes);
			break;
		case DispatcherFunction::DAY_FUNC:
			gpuDispatchPtr[i] = add_gpu_day_function(dataTypes);
			break;
		case DispatcherFunction::HOUR_FUNC:
			gpuDispatchPtr[i] = add_gpu_hour_function(dataTypes);
			break;
		case DispatcherFunction::MINUTE_FUNC:
			gpuDispatchPtr[i] = add_gpu_minute_function(dataTypes);
			break;
		case DispatcherFunction::SECOND_FUNC:
			gpuDispatchPtr[i] = add_gpu_second_function(dataTypes);
			break;
		case DispatcherFunction::ABS_FUNC:
			gpuDispatchPtr[i] = add_gpu_absolute_function(dataTypes);
			break;
		case DispatcherFunction::SIN_FUNC:
			gpuDispatchPtr[i] = add_gpu_sine_function(dataTypes);
			break;
		case DispatcherFunction::COS_FUNC:
			gpuDispatchPtr[i] = add_gpu_cosine_function(dataTypes);
			break;
		case DispatcherFunction::TAN_FUNC:
			gpuDispatchPtr[i] = add_gpu_tangent_function(dataTypes);
			break;
		case DispatcherFunction::COT_FUNC:
			gpuDispatchPtr[i] = add_gpu_cotangent_function(dataTypes);
			break;
		case DispatcherFunction::ASIN_FUNC:
			gpuDispatchPtr[i] = add_gpu_arcsine_function(dataTypes);
			break;
		case DispatcherFunction::ACOS_FUNC:
			gpuDispatchPtr[i] = add_gpu_arccosine_function(dataTypes);
			break;
		case DispatcherFunction::ATAN_FUNC:
			gpuDispatchPtr[i] = add_gpu_arctangent_function(dataTypes);
			break;
		case DispatcherFunction::LOG10_FUNC:
			gpuDispatchPtr[i] = add_gpu_logarithm10_function(dataTypes);
			break;
		case DispatcherFunction::LOG_FUNC:
			gpuDispatchPtr[i] = add_gpu_logarithmNatural_function(dataTypes);
			break;
		case DispatcherFunction::EXP_FUNC:
			gpuDispatchPtr[i] = add_gpu_exponential_function(dataTypes);
			break;
		case DispatcherFunction::SQRT_FUNC:
			gpuDispatchPtr[i] = add_gpu_squareRoot_function(dataTypes);
			break;
		case DispatcherFunction::SQUARE_FUNC:
			gpuDispatchPtr[i] = add_gpu_square_function(dataTypes);
			break;
		case DispatcherFunction::SIGN_FUNC:
			gpuDispatchPtr[i] = add_gpu_sign_function(dataTypes);
			break;
		case DispatcherFunction::ROUND_FUNC:
			gpuDispatchPtr[i] = add_gpu_round_function(dataTypes);
			break;
		case DispatcherFunction::FLOOR_FUNC:
			gpuDispatchPtr[i] = add_gpu_floor_function(dataTypes);
			break;
		case DispatcherFunction::CEIL_FUNC:
			gpuDispatchPtr[i] = add_gpu_ceil_function(dataTypes);
			break;
		case DispatcherFunction::PUSH_FUNC:
			gpuDispatchPtr[i] = add_gpu_push_function(i % DataType::DATA_TYPE_SIZE);
			break;
		default:
			gpuDispatchPtr[i] = reinterpret_cast<GpuVMFunction>(0xcccccccccccccccc);
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
	case 8:
		return &filterFunction<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::greater, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::greater, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::greater, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::greater, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::greater, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::greater, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::greater, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::greater, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::greater, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::greater, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::greater, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::greater, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::greater, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::greater, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<FilterConditions::less, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::less, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::less, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::less, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::less, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::less, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::less, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::less, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::less, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::less, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::less, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::less, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::less, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::less, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::less, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::less, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::less, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::less, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::less, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::greaterEqual, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::greaterEqual, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::greaterEqual, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::greaterEqual, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::lessEqual, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::lessEqual, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::lessEqual, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::lessEqual, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::equal, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::equal, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::equal, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::equal, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::equal, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::equal, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::equal, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::equal, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::equal, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::equal, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::equal, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::equal, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::equal, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::equal, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<FilterConditions::notEqual, int64_t, float>;
		break;
	case 11:
		return &filterFunction<FilterConditions::notEqual, int64_t, double>;
		break;
	case 15:
		return &filterFunction<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<FilterConditions::notEqual, float, int32_t>;
		break;
	case 17:
		return &filterFunction<FilterConditions::notEqual, float, int64_t>;
		break;
	case 18:
		return &filterFunction<FilterConditions::notEqual, float, float>;
		break;
	case 19:
		return &filterFunction<FilterConditions::notEqual, float, double>;
		break;
	case 23:
		return &filterFunction<FilterConditions::notEqual, float, int8_t>;
		break;
	case 24:
		return &filterFunction<FilterConditions::notEqual, double, int32_t>;
		break;
	case 25:
		return &filterFunction<FilterConditions::notEqual, double, int64_t>;
		break;
	case 26:
		return &filterFunction<FilterConditions::notEqual, double, float>;
		break;
	case 27:
		return &filterFunction<FilterConditions::notEqual, double, double>;
		break;
	case 31:
		return &filterFunction<FilterConditions::notEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunction<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<FilterConditions::notEqual, int8_t, float>;
		break;
	case 59:
		return &filterFunction<FilterConditions::notEqual, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, float>;
		break;
	case 11:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, double>;
		break;
	case 15:
		return &filterFunction<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<LogicOperations::logicalAnd, float, int32_t>;
		break;
	case 17:
		return &filterFunction<LogicOperations::logicalAnd, float, int64_t>;
		break;
	case 18:
		return &filterFunction<LogicOperations::logicalAnd, float, float>;
		break;
	case 19:
		return &filterFunction<LogicOperations::logicalAnd, float, double>;
		break;
	case 23:
		return &filterFunction<LogicOperations::logicalAnd, float, int8_t>;
		break;
	case 24:
		return &filterFunction<LogicOperations::logicalAnd, double, int32_t>;
		break;
	case 25:
		return &filterFunction<LogicOperations::logicalAnd, double, int64_t>;
		break;
	case 26:
		return &filterFunction<LogicOperations::logicalAnd, double, float>;
		break;
	case 27:
		return &filterFunction<LogicOperations::logicalAnd, double, double>;
		break;
	case 31:
		return &filterFunction<LogicOperations::logicalAnd, double, int8_t>;
		break;
	case 56:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, float>;
		break;
	case 59:
		return &filterFunction<LogicOperations::logicalAnd, int8_t, double>;
		break;
	case 63:
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
	case 8:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
	case 9:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
	case 10:
		return &filterFunction<LogicOperations::logicalOr, int64_t, float>;
		break;
	case 11:
		return &filterFunction<LogicOperations::logicalOr, int64_t, double>;
		break;
	case 15:
		return &filterFunction<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
	case 16:
		return &filterFunction<LogicOperations::logicalOr, float, int32_t>;
		break;
	case 17:
		return &filterFunction<LogicOperations::logicalOr, float, int64_t>;
		break;
	case 18:
		return &filterFunction<LogicOperations::logicalOr, float, float>;
		break;
	case 19:
		return &filterFunction<LogicOperations::logicalOr, float, double>;
		break;
	case 23:
		return &filterFunction<LogicOperations::logicalOr, float, int8_t>;
		break;
	case 24:
		return &filterFunction<LogicOperations::logicalOr, double, int32_t>;
		break;
	case 25:
		return &filterFunction<LogicOperations::logicalOr, double, int64_t>;
		break;
	case 26:
		return &filterFunction<LogicOperations::logicalOr, double, float>;
		break;
	case 27:
		return &filterFunction<LogicOperations::logicalOr, double, double>;
		break;
	case 31:
		return &filterFunction<LogicOperations::logicalOr, double, int8_t>;
		break;
	case 56:
		return &filterFunction<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
	case 57:
		return &filterFunction<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
	case 58:
		return &filterFunction<LogicOperations::logicalOr, int8_t, float>;
		break;
	case 59:
		return &filterFunction<LogicOperations::logicalOr, int8_t, double>;
		break;
	case 63:
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
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::mul, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::mul, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::mul, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::mul, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::mul, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::mul, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::mul, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::mul, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::mul, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::mul, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::mul, double, int8_t, double>;
		break;
	case 63:
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
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::div, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::div, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::div, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::div, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::div, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::div, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::div, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::div, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::div, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::div, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::div, double, int8_t, double>;
		break;
	case 63:
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
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::add, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::add, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::add, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::add, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::add, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::add, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::add, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::add, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::add, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::add, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::add, double, int8_t, double>;
		break;
	case 63:
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
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::sub, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::sub, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::sub, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::sub, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::sub, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::sub, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::sub, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::sub, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::sub, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::sub, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::sub, double, int8_t, double>;
		break;
	case 63:
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
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::mod, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::mod, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::mod, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::mod, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::mod>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_bitwiseOr_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::bitwiseOr, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::bitwiseOr>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_bitwiseAnd_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::bitwiseAnd, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::bitwiseAnd>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_bitwiseXor_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::bitwiseXor, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::bitwiseXor>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_bitwiseLeftShift_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::bitwiseLeftShift, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::bitwiseLeftShift>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_bitwiseRightShift_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int64_t, int32_t, int64_t>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int64_t, int64_t, int64_t>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int64_t, int64_t, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int64_t, int8_t, int64_t>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::bitwiseRightShift, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::bitwiseRightShift>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_logarithm_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::logarithm, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::logarithm, double, int8_t, double>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::logarithm, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::logarithm>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_power_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::power, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::power, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::power, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::power, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::power, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::power, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::power, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::power, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::power, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::power, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::power, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::power, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::power, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::power, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::power, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::power, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::power, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::power, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::power, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::power, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::power, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::power, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::power, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::power, double, int8_t, double>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::power, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::power>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_root_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::root, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::root, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::root, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::root, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::root, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::root, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::root, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::root, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::root, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::root, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::root, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::root, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::root, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::root, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::root, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::root, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::root, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::root, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::root, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::root, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::root, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::root, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::root, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::root, double, int8_t, double>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::root, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::root>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_arctangent2_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int32_t, int32_t, int32_t>;
		break;
	case 1:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int64_t, int32_t, int64_t>;
		break;
	case 2:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, int32_t, float>;
		break;
	case 3:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, int32_t, double>;
		break;
	case 7:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int32_t, int32_t, int8_t>;
		break;
	case 8:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int64_t, int64_t, int32_t>;
		break;
	case 9:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int64_t, int64_t, int64_t>;
		break;
	case 10:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, int64_t, float>;
		break;
	case 11:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, int64_t, double>;
		break;
	case 15:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int64_t, int64_t, int8_t>;
		break;
	case 16:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, float, int32_t>;
		break;
	case 17:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, float, int64_t>;
		break;
	case 18:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, float, float>;
		break;
	case 19:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, float, double>;
		break;
	case 23:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, float, int8_t>;
		break;
	case 24:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, double, int32_t>;
		break;
	case 25:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, double, int64_t>;
		break;
	case 26:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, double, float>;
		break;
	case 27:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, double, double>;
		break;
	case 31:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, double, int8_t>;
		break;
	case 56:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int32_t, int8_t, int32_t>;
		break;
	case 57:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int64_t, int8_t, int64_t>;
		break;
	case 58:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, float, int8_t, float>;
		break;
	case 59:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, double, int8_t, double>;
		break;
	case 63:
		return &arithmeticFunction<ArithmeticOperations::arctangent2, int8_t, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticOperations::arctangent2>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_minus_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::minus, std::conditional<std::is_same<ArithmeticUnaryOperations::minus::retType, void>::value, int32_t, ArithmeticUnaryOperations::minus::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::minus, std::conditional<std::is_same<ArithmeticUnaryOperations::minus::retType, void>::value, int64_t, ArithmeticUnaryOperations::minus::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::minus, std::conditional<std::is_same<ArithmeticUnaryOperations::minus::retType, void>::value, float, ArithmeticUnaryOperations::minus::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::minus, std::conditional<std::is_same<ArithmeticUnaryOperations::minus::retType, void>::value, double, ArithmeticUnaryOperations::minus::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::minus, std::conditional<std::is_same<ArithmeticUnaryOperations::minus::retType, void>::value, int8_t, ArithmeticUnaryOperations::minus::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::minus>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_absolute_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::absolute, std::conditional<std::is_same<ArithmeticUnaryOperations::absolute::retType, void>::value, int32_t, ArithmeticUnaryOperations::absolute::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::absolute, std::conditional<std::is_same<ArithmeticUnaryOperations::absolute::retType, void>::value, int64_t, ArithmeticUnaryOperations::absolute::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::absolute, std::conditional<std::is_same<ArithmeticUnaryOperations::absolute::retType, void>::value, float, ArithmeticUnaryOperations::absolute::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::absolute, std::conditional<std::is_same<ArithmeticUnaryOperations::absolute::retType, void>::value, double, ArithmeticUnaryOperations::absolute::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::absolute, std::conditional<std::is_same<ArithmeticUnaryOperations::absolute::retType, void>::value, int8_t, ArithmeticUnaryOperations::absolute::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::absolute>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_sine_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sine, std::conditional<std::is_same<ArithmeticUnaryOperations::sine::retType, void>::value, int32_t, ArithmeticUnaryOperations::sine::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sine, std::conditional<std::is_same<ArithmeticUnaryOperations::sine::retType, void>::value, int64_t, ArithmeticUnaryOperations::sine::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sine, std::conditional<std::is_same<ArithmeticUnaryOperations::sine::retType, void>::value, float, ArithmeticUnaryOperations::sine::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sine, std::conditional<std::is_same<ArithmeticUnaryOperations::sine::retType, void>::value, double, ArithmeticUnaryOperations::sine::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sine, std::conditional<std::is_same<ArithmeticUnaryOperations::sine::retType, void>::value, int8_t, ArithmeticUnaryOperations::sine::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::sine>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_cosine_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cosine, std::conditional<std::is_same<ArithmeticUnaryOperations::cosine::retType, void>::value, int32_t, ArithmeticUnaryOperations::cosine::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cosine, std::conditional<std::is_same<ArithmeticUnaryOperations::cosine::retType, void>::value, int64_t, ArithmeticUnaryOperations::cosine::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cosine, std::conditional<std::is_same<ArithmeticUnaryOperations::cosine::retType, void>::value, float, ArithmeticUnaryOperations::cosine::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cosine, std::conditional<std::is_same<ArithmeticUnaryOperations::cosine::retType, void>::value, double, ArithmeticUnaryOperations::cosine::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cosine, std::conditional<std::is_same<ArithmeticUnaryOperations::cosine::retType, void>::value, int8_t, ArithmeticUnaryOperations::cosine::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::cosine>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_tangent_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::tangent, std::conditional<std::is_same<ArithmeticUnaryOperations::tangent::retType, void>::value, int32_t, ArithmeticUnaryOperations::tangent::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::tangent, std::conditional<std::is_same<ArithmeticUnaryOperations::tangent::retType, void>::value, int64_t, ArithmeticUnaryOperations::tangent::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::tangent, std::conditional<std::is_same<ArithmeticUnaryOperations::tangent::retType, void>::value, float, ArithmeticUnaryOperations::tangent::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::tangent, std::conditional<std::is_same<ArithmeticUnaryOperations::tangent::retType, void>::value, double, ArithmeticUnaryOperations::tangent::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::tangent, std::conditional<std::is_same<ArithmeticUnaryOperations::tangent::retType, void>::value, int8_t, ArithmeticUnaryOperations::tangent::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::tangent>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_cotangent_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cotangent, std::conditional<std::is_same<ArithmeticUnaryOperations::cotangent::retType, void>::value, int32_t, ArithmeticUnaryOperations::cotangent::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cotangent, std::conditional<std::is_same<ArithmeticUnaryOperations::cotangent::retType, void>::value, int64_t, ArithmeticUnaryOperations::cotangent::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cotangent, std::conditional<std::is_same<ArithmeticUnaryOperations::cotangent::retType, void>::value, float, ArithmeticUnaryOperations::cotangent::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cotangent, std::conditional<std::is_same<ArithmeticUnaryOperations::cotangent::retType, void>::value, double, ArithmeticUnaryOperations::cotangent::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::cotangent, std::conditional<std::is_same<ArithmeticUnaryOperations::cotangent::retType, void>::value, int8_t, ArithmeticUnaryOperations::cotangent::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::cotangent>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_arcsine_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arcsine, std::conditional<std::is_same<ArithmeticUnaryOperations::arcsine::retType, void>::value, int32_t, ArithmeticUnaryOperations::arcsine::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arcsine, std::conditional<std::is_same<ArithmeticUnaryOperations::arcsine::retType, void>::value, int64_t, ArithmeticUnaryOperations::arcsine::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arcsine, std::conditional<std::is_same<ArithmeticUnaryOperations::arcsine::retType, void>::value, float, ArithmeticUnaryOperations::arcsine::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arcsine, std::conditional<std::is_same<ArithmeticUnaryOperations::arcsine::retType, void>::value, double, ArithmeticUnaryOperations::arcsine::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arcsine, std::conditional<std::is_same<ArithmeticUnaryOperations::arcsine::retType, void>::value, int8_t, ArithmeticUnaryOperations::arcsine::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::arcsine>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_arccosine_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arccosine, std::conditional<std::is_same<ArithmeticUnaryOperations::arccosine::retType, void>::value, int32_t, ArithmeticUnaryOperations::arccosine::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arccosine, std::conditional<std::is_same<ArithmeticUnaryOperations::arccosine::retType, void>::value, int64_t, ArithmeticUnaryOperations::arccosine::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arccosine, std::conditional<std::is_same<ArithmeticUnaryOperations::arccosine::retType, void>::value, float, ArithmeticUnaryOperations::arccosine::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arccosine, std::conditional<std::is_same<ArithmeticUnaryOperations::arccosine::retType, void>::value, double, ArithmeticUnaryOperations::arccosine::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arccosine, std::conditional<std::is_same<ArithmeticUnaryOperations::arccosine::retType, void>::value, int8_t, ArithmeticUnaryOperations::arccosine::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::arccosine>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_arctangent_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arctangent, std::conditional<std::is_same<ArithmeticUnaryOperations::arctangent::retType, void>::value, int32_t, ArithmeticUnaryOperations::arctangent::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arctangent, std::conditional<std::is_same<ArithmeticUnaryOperations::arctangent::retType, void>::value, int64_t, ArithmeticUnaryOperations::arctangent::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arctangent, std::conditional<std::is_same<ArithmeticUnaryOperations::arctangent::retType, void>::value, float, ArithmeticUnaryOperations::arctangent::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arctangent, std::conditional<std::is_same<ArithmeticUnaryOperations::arctangent::retType, void>::value, double, ArithmeticUnaryOperations::arctangent::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::arctangent, std::conditional<std::is_same<ArithmeticUnaryOperations::arctangent::retType, void>::value, int8_t, ArithmeticUnaryOperations::arctangent::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::arctangent>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_logarithm10_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithm10, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithm10::retType, void>::value, int32_t, ArithmeticUnaryOperations::logarithm10::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithm10, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithm10::retType, void>::value, int64_t, ArithmeticUnaryOperations::logarithm10::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithm10, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithm10::retType, void>::value, float, ArithmeticUnaryOperations::logarithm10::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithm10, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithm10::retType, void>::value, double, ArithmeticUnaryOperations::logarithm10::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithm10, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithm10::retType, void>::value, int8_t, ArithmeticUnaryOperations::logarithm10::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::logarithm10>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_logarithmNatural_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithmNatural, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithmNatural::retType, void>::value, int32_t, ArithmeticUnaryOperations::logarithmNatural::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithmNatural, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithmNatural::retType, void>::value, int64_t, ArithmeticUnaryOperations::logarithmNatural::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithmNatural, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithmNatural::retType, void>::value, float, ArithmeticUnaryOperations::logarithmNatural::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithmNatural, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithmNatural::retType, void>::value, double, ArithmeticUnaryOperations::logarithmNatural::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::logarithmNatural, std::conditional<std::is_same<ArithmeticUnaryOperations::logarithmNatural::retType, void>::value, int8_t, ArithmeticUnaryOperations::logarithmNatural::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::logarithmNatural>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_exponential_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::exponential, std::conditional<std::is_same<ArithmeticUnaryOperations::exponential::retType, void>::value, int32_t, ArithmeticUnaryOperations::exponential::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::exponential, std::conditional<std::is_same<ArithmeticUnaryOperations::exponential::retType, void>::value, int64_t, ArithmeticUnaryOperations::exponential::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::exponential, std::conditional<std::is_same<ArithmeticUnaryOperations::exponential::retType, void>::value, float, ArithmeticUnaryOperations::exponential::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::exponential, std::conditional<std::is_same<ArithmeticUnaryOperations::exponential::retType, void>::value, double, ArithmeticUnaryOperations::exponential::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::exponential, std::conditional<std::is_same<ArithmeticUnaryOperations::exponential::retType, void>::value, int8_t, ArithmeticUnaryOperations::exponential::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::exponential>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_squareRoot_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::squareRoot, std::conditional<std::is_same<ArithmeticUnaryOperations::squareRoot::retType, void>::value, int32_t, ArithmeticUnaryOperations::squareRoot::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::squareRoot, std::conditional<std::is_same<ArithmeticUnaryOperations::squareRoot::retType, void>::value, int64_t, ArithmeticUnaryOperations::squareRoot::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::squareRoot, std::conditional<std::is_same<ArithmeticUnaryOperations::squareRoot::retType, void>::value, float, ArithmeticUnaryOperations::squareRoot::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::squareRoot, std::conditional<std::is_same<ArithmeticUnaryOperations::squareRoot::retType, void>::value, double, ArithmeticUnaryOperations::squareRoot::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::squareRoot, std::conditional<std::is_same<ArithmeticUnaryOperations::squareRoot::retType, void>::value, int8_t, ArithmeticUnaryOperations::squareRoot::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::squareRoot>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_square_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::square, std::conditional<std::is_same<ArithmeticUnaryOperations::square::retType, void>::value, int32_t, ArithmeticUnaryOperations::square::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::square, std::conditional<std::is_same<ArithmeticUnaryOperations::square::retType, void>::value, int64_t, ArithmeticUnaryOperations::square::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::square, std::conditional<std::is_same<ArithmeticUnaryOperations::square::retType, void>::value, float, ArithmeticUnaryOperations::square::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::square, std::conditional<std::is_same<ArithmeticUnaryOperations::square::retType, void>::value, double, ArithmeticUnaryOperations::square::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::square, std::conditional<std::is_same<ArithmeticUnaryOperations::square::retType, void>::value, int8_t, ArithmeticUnaryOperations::square::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::square>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_sign_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sign, std::conditional<std::is_same<ArithmeticUnaryOperations::sign::retType, void>::value, int32_t, ArithmeticUnaryOperations::sign::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sign, std::conditional<std::is_same<ArithmeticUnaryOperations::sign::retType, void>::value, int64_t, ArithmeticUnaryOperations::sign::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sign, std::conditional<std::is_same<ArithmeticUnaryOperations::sign::retType, void>::value, float, ArithmeticUnaryOperations::sign::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sign, std::conditional<std::is_same<ArithmeticUnaryOperations::sign::retType, void>::value, double, ArithmeticUnaryOperations::sign::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::sign, std::conditional<std::is_same<ArithmeticUnaryOperations::sign::retType, void>::value, int8_t, ArithmeticUnaryOperations::sign::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::sign>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_round_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::round, std::conditional<std::is_same<ArithmeticUnaryOperations::round::retType, void>::value, int32_t, ArithmeticUnaryOperations::round::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::round, std::conditional<std::is_same<ArithmeticUnaryOperations::round::retType, void>::value, int64_t, ArithmeticUnaryOperations::round::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::round, std::conditional<std::is_same<ArithmeticUnaryOperations::round::retType, void>::value, float, ArithmeticUnaryOperations::round::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::round, std::conditional<std::is_same<ArithmeticUnaryOperations::round::retType, void>::value, double, ArithmeticUnaryOperations::round::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::round, std::conditional<std::is_same<ArithmeticUnaryOperations::round::retType, void>::value, int8_t, ArithmeticUnaryOperations::round::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::round>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_floor_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::floor, std::conditional<std::is_same<ArithmeticUnaryOperations::floor::retType, void>::value, int32_t, ArithmeticUnaryOperations::floor::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::floor, std::conditional<std::is_same<ArithmeticUnaryOperations::floor::retType, void>::value, int64_t, ArithmeticUnaryOperations::floor::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::floor, std::conditional<std::is_same<ArithmeticUnaryOperations::floor::retType, void>::value, float, ArithmeticUnaryOperations::floor::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::floor, std::conditional<std::is_same<ArithmeticUnaryOperations::floor::retType, void>::value, double, ArithmeticUnaryOperations::floor::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::floor, std::conditional<std::is_same<ArithmeticUnaryOperations::floor::retType, void>::value, int8_t, ArithmeticUnaryOperations::floor::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::floor>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_ceil_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::ceil, std::conditional<std::is_same<ArithmeticUnaryOperations::ceil::retType, void>::value, int32_t, ArithmeticUnaryOperations::ceil::retType>::type, int32_t>;
		break;
	case 1:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::ceil, std::conditional<std::is_same<ArithmeticUnaryOperations::ceil::retType, void>::value, int64_t, ArithmeticUnaryOperations::ceil::retType>::type, int64_t>;
		break;
	case 2:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::ceil, std::conditional<std::is_same<ArithmeticUnaryOperations::ceil::retType, void>::value, float, ArithmeticUnaryOperations::ceil::retType>::type, float>;
		break;
	case 3:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::ceil, std::conditional<std::is_same<ArithmeticUnaryOperations::ceil::retType, void>::value, double, ArithmeticUnaryOperations::ceil::retType>::type, double>;
		break;
	case 7:
		return &arithmeticUnaryFunction<ArithmeticUnaryOperations::ceil, std::conditional<std::is_same<ArithmeticUnaryOperations::ceil::retType, void>::value, int8_t, ArithmeticUnaryOperations::ceil::retType>::type, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<ArithmeticUnaryOperations::ceil>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_year_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::year>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::year>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_month_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::month>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::month>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_day_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::day>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::day>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_hour_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::hour>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::hour>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_minute_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::minute>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::minute>;
		break;
	}
}


__device__ GpuVMFunction add_gpu_second_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 1:
		return &dateFunction<DateOperations::second>;
		break;
	default:
		return &invalidArgumentTypeHandler<DateOperations::second>;
		break;
	}
}

__device__ GpuVMFunction add_gpu_logicalNot_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &logicalNotFunction<LogicOperations::logicalNot, int32_t>;
		break;
	case 1:
		return &logicalNotFunction<LogicOperations::logicalNot, int64_t>;
		break;
	case 2:
		return &logicalNotFunction<LogicOperations::logicalNot, float>;
		break;
	case 3:
		return &logicalNotFunction<LogicOperations::logicalNot, double>;
		break;
	case 7:
		return &logicalNotFunction<LogicOperations::logicalNot, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<LogicOperations::logicalNot>;
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