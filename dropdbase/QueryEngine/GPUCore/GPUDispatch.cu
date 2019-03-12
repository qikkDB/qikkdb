#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../Context.h"
#include "../../DataType.h"
#include "GPUFilter.cuh"
#include "GPULogic.cuh"
#include "GPUDispatch.cuh"
#include "MaybeDeref.cuh"


__device__ DispatchFunction add_gpu_greater_function(int32_t dataTypes)
{
	switch (dataTypes)
	{
	case 0:
		return &filterFunctionConstConst<FilterConditions::greater, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::greater, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::greater, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::greater, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::greater, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::greater, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::greater, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::greater, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::greater, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::greater, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::greater, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::greater, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::greater, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::greater, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::greater, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::greater, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::greater, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::greater, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::greater, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::greater, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::greater, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::greater, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::greater, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::greater, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::greater, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::greater, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::greater, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::greater, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::greater, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::greater, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::greater, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::greater, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::greater, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::greater, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::greater, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::greater, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::greater, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::greater, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::greater, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::greater, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::greater, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::greater, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::greater, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::greater, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::greater, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::greater, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::greater, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::greater, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::greater, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::greater, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::greater, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::greater, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::greater, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::greater, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::greater, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::greater, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::greater, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::greater, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::greater, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::greater, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::greater, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::greater, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::greater, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::greater, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::greater, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::greater, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::greater, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::greater, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::greater, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::greater, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::greater, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::greater, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::greater, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::greater, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::greater, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::greater, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::greater, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::greater, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::greater, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::greater, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::greater, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::greater, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::greater, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::greater, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::greater, int8_t, int8_t>;
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
		return &filterFunctionConstConst<FilterConditions::less, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::less, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::less, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::less, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::less, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::less, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::less, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::less, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::less, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::less, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::less, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::less, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::less, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::less, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::less, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::less, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::less, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::less, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::less, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::less, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::less, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::less, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::less, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::less, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::less, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::less, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::less, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::less, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::less, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::less, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::less, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::less, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::less, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::less, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::less, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::less, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::less, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::less, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::less, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::less, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::less, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::less, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::less, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::less, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::less, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::less, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::less, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::less, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::less, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::less, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::less, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::less, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::less, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::less, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::less, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::less, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::less, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::less, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::less, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::less, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::less, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::less, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::less, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::less, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::less, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::less, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::less, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::less, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::less, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::less, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::less, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::less, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::less, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::less, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::less, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::less, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::less, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::less, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::less, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::less, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::less, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::less, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::less, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::less, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::less, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::less, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::less, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::less, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::less, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::less, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::less, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::less, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::less, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::less, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::less, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::less, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::less, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::less, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::less, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::less, int8_t, int8_t>;
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
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::greaterEqual, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::greaterEqual, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::greaterEqual, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::greaterEqual, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::greaterEqual, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::greaterEqual, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::greaterEqual, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::greaterEqual, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::greaterEqual, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::greaterEqual, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::greaterEqual, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::greaterEqual, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::greaterEqual, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::greaterEqual, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::greaterEqual, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::greaterEqual, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::greaterEqual, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::greaterEqual, int8_t, int8_t>;
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
		return &filterFunctionConstConst<FilterConditions::lessEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::lessEqual, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::lessEqual, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::lessEqual, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::lessEqual, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::lessEqual, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::lessEqual, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::lessEqual, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::lessEqual, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::lessEqual, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::lessEqual, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::lessEqual, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::lessEqual, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::lessEqual, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::lessEqual, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::lessEqual, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::lessEqual, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::lessEqual, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::lessEqual, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::lessEqual, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::lessEqual, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::lessEqual, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::lessEqual, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::lessEqual, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::lessEqual, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::lessEqual, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::lessEqual, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::lessEqual, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::lessEqual, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::lessEqual, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::lessEqual, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::lessEqual, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::lessEqual, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::lessEqual, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::lessEqual, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::lessEqual, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::lessEqual, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::lessEqual, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::lessEqual, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::lessEqual, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::lessEqual, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::lessEqual, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::lessEqual, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::lessEqual, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::lessEqual, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::lessEqual, int8_t, int8_t>;
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
		return &filterFunctionConstConst<FilterConditions::equal, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::equal, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::equal, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::equal, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::equal, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::equal, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::equal, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::equal, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::equal, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::equal, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::equal, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::equal, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::equal, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::equal, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::equal, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::equal, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::equal, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::equal, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::equal, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::equal, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::equal, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::equal, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::equal, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::equal, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::equal, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::equal, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::equal, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::equal, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::equal, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::equal, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::equal, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::equal, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::equal, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::equal, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::equal, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::equal, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::equal, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::equal, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::equal, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::equal, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::equal, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::equal, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::equal, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::equal, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::equal, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::equal, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::equal, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::equal, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::equal, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::equal, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::equal, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::equal, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::equal, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::equal, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::equal, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::equal, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::equal, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::equal, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::equal, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::equal, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::equal, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::equal, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::equal, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::equal, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::equal, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::equal, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::equal, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::equal, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::equal, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::equal, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::equal, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::equal, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::equal, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::equal, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::equal, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::equal, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::equal, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::equal, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::equal, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::equal, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::equal, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::equal, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::equal, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::equal, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::equal, int8_t, int8_t>;
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
		return &filterFunctionConstConst<FilterConditions::notEqual, int32_t, int32_t>;
		break;
	case 1:
		return &filterFunctionConstConst<FilterConditions::notEqual, int32_t, int64_t>;
		break;
	case 2:
		return &filterFunctionConstConst<FilterConditions::notEqual, int32_t, float>;
		break;
	case 3:
		return &filterFunctionConstConst<FilterConditions::notEqual, int32_t, double>;
		break;
	case 7:
		return &filterFunctionConstConst<FilterConditions::notEqual, int32_t, int8_t>;
		break;
	case 8:
		return &filterFunctionConstCol<FilterConditions::notEqual, int32_t, int32_t>;
		break;
	case 9:
		return &filterFunctionConstCol<FilterConditions::notEqual, int32_t, int64_t>;
		break;
	case 10:
		return &filterFunctionConstCol<FilterConditions::notEqual, int32_t, float>;
		break;
	case 11:
		return &filterFunctionConstCol<FilterConditions::notEqual, int32_t, double>;
		break;
	case 15:
		return &filterFunctionConstCol<FilterConditions::notEqual, int32_t, int8_t>;
		break;
	case 16:
		return &filterFunctionConstConst<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 17:
		return &filterFunctionConstConst<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 18:
		return &filterFunctionConstConst<FilterConditions::notEqual, int64_t, float>;
		break;
	case 19:
		return &filterFunctionConstConst<FilterConditions::notEqual, int64_t, double>;
		break;
	case 23:
		return &filterFunctionConstConst<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 24:
		return &filterFunctionConstCol<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 25:
		return &filterFunctionConstCol<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 26:
		return &filterFunctionConstCol<FilterConditions::notEqual, int64_t, float>;
		break;
	case 27:
		return &filterFunctionConstCol<FilterConditions::notEqual, int64_t, double>;
		break;
	case 31:
		return &filterFunctionConstCol<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 32:
		return &filterFunctionConstConst<FilterConditions::notEqual, float, int32_t>;
		break;
	case 33:
		return &filterFunctionConstConst<FilterConditions::notEqual, float, int64_t>;
		break;
	case 34:
		return &filterFunctionConstConst<FilterConditions::notEqual, float, float>;
		break;
	case 35:
		return &filterFunctionConstConst<FilterConditions::notEqual, float, double>;
		break;
	case 39:
		return &filterFunctionConstConst<FilterConditions::notEqual, float, int8_t>;
		break;
	case 40:
		return &filterFunctionConstCol<FilterConditions::notEqual, float, int32_t>;
		break;
	case 41:
		return &filterFunctionConstCol<FilterConditions::notEqual, float, int64_t>;
		break;
	case 42:
		return &filterFunctionConstCol<FilterConditions::notEqual, float, float>;
		break;
	case 43:
		return &filterFunctionConstCol<FilterConditions::notEqual, float, double>;
		break;
	case 47:
		return &filterFunctionConstCol<FilterConditions::notEqual, float, int8_t>;
		break;
	case 48:
		return &filterFunctionConstConst<FilterConditions::notEqual, double, int32_t>;
		break;
	case 49:
		return &filterFunctionConstConst<FilterConditions::notEqual, double, int64_t>;
		break;
	case 50:
		return &filterFunctionConstConst<FilterConditions::notEqual, double, float>;
		break;
	case 51:
		return &filterFunctionConstConst<FilterConditions::notEqual, double, double>;
		break;
	case 55:
		return &filterFunctionConstConst<FilterConditions::notEqual, double, int8_t>;
		break;
	case 56:
		return &filterFunctionConstCol<FilterConditions::notEqual, double, int32_t>;
		break;
	case 57:
		return &filterFunctionConstCol<FilterConditions::notEqual, double, int64_t>;
		break;
	case 58:
		return &filterFunctionConstCol<FilterConditions::notEqual, double, float>;
		break;
	case 59:
		return &filterFunctionConstCol<FilterConditions::notEqual, double, double>;
		break;
	case 63:
		return &filterFunctionConstCol<FilterConditions::notEqual, double, int8_t>;
		break;
	case 112:
		return &filterFunctionConstConst<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 113:
		return &filterFunctionConstConst<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 114:
		return &filterFunctionConstConst<FilterConditions::notEqual, int8_t, float>;
		break;
	case 115:
		return &filterFunctionConstConst<FilterConditions::notEqual, int8_t, double>;
		break;
	case 119:
		return &filterFunctionConstConst<FilterConditions::notEqual, int8_t, int8_t>;
		break;
	case 120:
		return &filterFunctionConstCol<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 121:
		return &filterFunctionConstCol<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 122:
		return &filterFunctionConstCol<FilterConditions::notEqual, int8_t, float>;
		break;
	case 123:
		return &filterFunctionConstCol<FilterConditions::notEqual, int8_t, double>;
		break;
	case 127:
		return &filterFunctionConstCol<FilterConditions::notEqual, int8_t, int8_t>;
		break;
	case 128:
		return &filterFunctionColConst<FilterConditions::notEqual, int32_t, int32_t>;
		break;
	case 129:
		return &filterFunctionColConst<FilterConditions::notEqual, int32_t, int64_t>;
		break;
	case 130:
		return &filterFunctionColConst<FilterConditions::notEqual, int32_t, float>;
		break;
	case 131:
		return &filterFunctionColConst<FilterConditions::notEqual, int32_t, double>;
		break;
	case 135:
		return &filterFunctionColConst<FilterConditions::notEqual, int32_t, int8_t>;
		break;
	case 136:
		return &filterFunctionColCol<FilterConditions::notEqual, int32_t, int32_t>;
		break;
	case 137:
		return &filterFunctionColCol<FilterConditions::notEqual, int32_t, int64_t>;
		break;
	case 138:
		return &filterFunctionColCol<FilterConditions::notEqual, int32_t, float>;
		break;
	case 139:
		return &filterFunctionColCol<FilterConditions::notEqual, int32_t, double>;
		break;
	case 143:
		return &filterFunctionColCol<FilterConditions::notEqual, int32_t, int8_t>;
		break;
	case 144:
		return &filterFunctionColConst<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 145:
		return &filterFunctionColConst<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 146:
		return &filterFunctionColConst<FilterConditions::notEqual, int64_t, float>;
		break;
	case 147:
		return &filterFunctionColConst<FilterConditions::notEqual, int64_t, double>;
		break;
	case 151:
		return &filterFunctionColConst<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 152:
		return &filterFunctionColCol<FilterConditions::notEqual, int64_t, int32_t>;
		break;
	case 153:
		return &filterFunctionColCol<FilterConditions::notEqual, int64_t, int64_t>;
		break;
	case 154:
		return &filterFunctionColCol<FilterConditions::notEqual, int64_t, float>;
		break;
	case 155:
		return &filterFunctionColCol<FilterConditions::notEqual, int64_t, double>;
		break;
	case 159:
		return &filterFunctionColCol<FilterConditions::notEqual, int64_t, int8_t>;
		break;
	case 160:
		return &filterFunctionColConst<FilterConditions::notEqual, float, int32_t>;
		break;
	case 161:
		return &filterFunctionColConst<FilterConditions::notEqual, float, int64_t>;
		break;
	case 162:
		return &filterFunctionColConst<FilterConditions::notEqual, float, float>;
		break;
	case 163:
		return &filterFunctionColConst<FilterConditions::notEqual, float, double>;
		break;
	case 167:
		return &filterFunctionColConst<FilterConditions::notEqual, float, int8_t>;
		break;
	case 168:
		return &filterFunctionColCol<FilterConditions::notEqual, float, int32_t>;
		break;
	case 169:
		return &filterFunctionColCol<FilterConditions::notEqual, float, int64_t>;
		break;
	case 170:
		return &filterFunctionColCol<FilterConditions::notEqual, float, float>;
		break;
	case 171:
		return &filterFunctionColCol<FilterConditions::notEqual, float, double>;
		break;
	case 175:
		return &filterFunctionColCol<FilterConditions::notEqual, float, int8_t>;
		break;
	case 176:
		return &filterFunctionColConst<FilterConditions::notEqual, double, int32_t>;
		break;
	case 177:
		return &filterFunctionColConst<FilterConditions::notEqual, double, int64_t>;
		break;
	case 178:
		return &filterFunctionColConst<FilterConditions::notEqual, double, float>;
		break;
	case 179:
		return &filterFunctionColConst<FilterConditions::notEqual, double, double>;
		break;
	case 183:
		return &filterFunctionColConst<FilterConditions::notEqual, double, int8_t>;
		break;
	case 184:
		return &filterFunctionColCol<FilterConditions::notEqual, double, int32_t>;
		break;
	case 185:
		return &filterFunctionColCol<FilterConditions::notEqual, double, int64_t>;
		break;
	case 186:
		return &filterFunctionColCol<FilterConditions::notEqual, double, float>;
		break;
	case 187:
		return &filterFunctionColCol<FilterConditions::notEqual, double, double>;
		break;
	case 191:
		return &filterFunctionColCol<FilterConditions::notEqual, double, int8_t>;
		break;
	case 240:
		return &filterFunctionColConst<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 241:
		return &filterFunctionColConst<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 242:
		return &filterFunctionColConst<FilterConditions::notEqual, int8_t, float>;
		break;
	case 243:
		return &filterFunctionColConst<FilterConditions::notEqual, int8_t, double>;
		break;
	case 247:
		return &filterFunctionColConst<FilterConditions::notEqual, int8_t, int8_t>;
		break;
	case 248:
		return &filterFunctionColCol<FilterConditions::notEqual, int8_t, int32_t>;
		break;
	case 249:
		return &filterFunctionColCol<FilterConditions::notEqual, int8_t, int64_t>;
		break;
	case 250:
		return &filterFunctionColCol<FilterConditions::notEqual, int8_t, float>;
		break;
	case 251:
		return &filterFunctionColCol<FilterConditions::notEqual, int8_t, double>;
		break;
	case 255:
		return &filterFunctionColCol<FilterConditions::notEqual, int8_t, int8_t>;
		break;
	default:
		return &invalidArgumentTypeHandler<FilterConditions::notEqual>;
		break;
	}
}

__device__ DispatchFunction add_gpu_logicalAnd_function(int32_t dataTypes)
{
	switch(dataTypes)
	{
		case 0:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int32_t, int32_t>;
		break;
		case 1:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int32_t, int64_t>;
		break;
		case 2:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int32_t, float>;
		break;
		case 3:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int32_t, double>;
		break;
		case 7:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int32_t, int8_t>;
		break;
		case 8:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int32_t, int32_t>;
		break;
		case 9:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int32_t, int64_t>;
		break;
		case 10:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int32_t, float>;
		break;
		case 11:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int32_t, double>;
		break;
		case 15:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int32_t, int8_t>;
		break;
		case 16:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
		case 17:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
		case 18:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int64_t, float>;
		break;
		case 19:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int64_t, double>;
		break;
		case 23:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
		case 24:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
		case 25:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
		case 26:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int64_t, float>;
		break;
		case 27:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int64_t, double>;
		break;
		case 31:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
		case 32:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, float, int32_t>;
		break;
		case 33:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, float, int64_t>;
		break;
		case 34:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, float, float>;
		break;
		case 35:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, float, double>;
		break;
		case 39:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, float, int8_t>;
		break;
		case 40:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, float, int32_t>;
		break;
		case 41:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, float, int64_t>;
		break;
		case 42:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, float, float>;
		break;
		case 43:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, float, double>;
		break;
		case 47:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, float, int8_t>;
		break;
		case 48:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, double, int32_t>;
		break;
		case 49:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, double, int64_t>;
		break;
		case 50:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, double, float>;
		break;
		case 51:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, double, double>;
		break;
		case 55:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, double, int8_t>;
		break;
		case 56:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, double, int32_t>;
		break;
		case 57:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, double, int64_t>;
		break;
		case 58:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, double, float>;
		break;
		case 59:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, double, double>;
		break;
		case 63:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, double, int8_t>;
		break;
		case 112:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
		case 113:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
		case 114:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int8_t, float>;
		break;
		case 115:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int8_t, double>;
		break;
		case 119:
			return &filterFunctionConstConst<LogicOperations::logicalAnd, int8_t, int8_t>;
		break;
		case 120:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
		case 121:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
		case 122:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int8_t, float>;
		break;
		case 123:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int8_t, double>;
		break;
		case 127:
			return &filterFunctionConstCol<LogicOperations::logicalAnd, int8_t, int8_t>;
		break;
		case 128:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int32_t, int32_t>;
		break;
		case 129:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int32_t, int64_t>;
		break;
		case 130:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int32_t, float>;
		break;
		case 131:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int32_t, double>;
		break;
		case 135:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int32_t, int8_t>;
		break;
		case 136:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int32_t, int32_t>;
		break;
		case 137:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int32_t, int64_t>;
		break;
		case 138:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int32_t, float>;
		break;
		case 139:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int32_t, double>;
		break;
		case 143:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int32_t, int8_t>;
		break;
		case 144:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
		case 145:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
		case 146:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int64_t, float>;
		break;
		case 147:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int64_t, double>;
		break;
		case 151:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
		case 152:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int64_t, int32_t>;
		break;
		case 153:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int64_t, int64_t>;
		break;
		case 154:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int64_t, float>;
		break;
		case 155:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int64_t, double>;
		break;
		case 159:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int64_t, int8_t>;
		break;
		case 160:
			return &filterFunctionColConst<LogicOperations::logicalAnd, float, int32_t>;
		break;
		case 161:
			return &filterFunctionColConst<LogicOperations::logicalAnd, float, int64_t>;
		break;
		case 162:
			return &filterFunctionColConst<LogicOperations::logicalAnd, float, float>;
		break;
		case 163:
			return &filterFunctionColConst<LogicOperations::logicalAnd, float, double>;
		break;
		case 167:
			return &filterFunctionColConst<LogicOperations::logicalAnd, float, int8_t>;
		break;
		case 168:
			return &filterFunctionColCol<LogicOperations::logicalAnd, float, int32_t>;
		break;
		case 169:
			return &filterFunctionColCol<LogicOperations::logicalAnd, float, int64_t>;
		break;
		case 170:
			return &filterFunctionColCol<LogicOperations::logicalAnd, float, float>;
		break;
		case 171:
			return &filterFunctionColCol<LogicOperations::logicalAnd, float, double>;
		break;
		case 175:
			return &filterFunctionColCol<LogicOperations::logicalAnd, float, int8_t>;
		break;
		case 176:
			return &filterFunctionColConst<LogicOperations::logicalAnd, double, int32_t>;
		break;
		case 177:
			return &filterFunctionColConst<LogicOperations::logicalAnd, double, int64_t>;
		break;
		case 178:
			return &filterFunctionColConst<LogicOperations::logicalAnd, double, float>;
		break;
		case 179:
			return &filterFunctionColConst<LogicOperations::logicalAnd, double, double>;
		break;
		case 183:
			return &filterFunctionColConst<LogicOperations::logicalAnd, double, int8_t>;
		break;
		case 184:
			return &filterFunctionColCol<LogicOperations::logicalAnd, double, int32_t>;
		break;
		case 185:
			return &filterFunctionColCol<LogicOperations::logicalAnd, double, int64_t>;
		break;
		case 186:
			return &filterFunctionColCol<LogicOperations::logicalAnd, double, float>;
		break;
		case 187:
			return &filterFunctionColCol<LogicOperations::logicalAnd, double, double>;
		break;
		case 191:
			return &filterFunctionColCol<LogicOperations::logicalAnd, double, int8_t>;
		break;
		case 240:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
		case 241:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
		case 242:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int8_t, float>;
		break;
		case 243:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int8_t, double>;
		break;
		case 247:
			return &filterFunctionColConst<LogicOperations::logicalAnd, int8_t, int8_t>;
		break;
		case 248:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int8_t, int32_t>;
		break;
		case 249:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int8_t, int64_t>;
		break;
		case 250:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int8_t, float>;
		break;
		case 251:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int8_t, double>;
		break;
		case 255:
			return &filterFunctionColCol<LogicOperations::logicalAnd, int8_t, int8_t>;
		break;
		default:
			return &invalidArgumentTypeHandler<LogicOperations::logicalAnd>;
		break;
	}
}


__device__ DispatchFunction add_gpu_logicalOr_function(int32_t dataTypes)
{
	switch(dataTypes)
	{
		case 0:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int32_t, int32_t>;
		break;
		case 1:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int32_t, int64_t>;
		break;
		case 2:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int32_t, float>;
		break;
		case 3:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int32_t, double>;
		break;
		case 7:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int32_t, int8_t>;
		break;
		case 8:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int32_t, int32_t>;
		break;
		case 9:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int32_t, int64_t>;
		break;
		case 10:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int32_t, float>;
		break;
		case 11:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int32_t, double>;
		break;
		case 15:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int32_t, int8_t>;
		break;
		case 16:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
		case 17:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
		case 18:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int64_t, float>;
		break;
		case 19:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int64_t, double>;
		break;
		case 23:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
		case 24:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
		case 25:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
		case 26:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int64_t, float>;
		break;
		case 27:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int64_t, double>;
		break;
		case 31:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
		case 32:
			return &filterFunctionConstConst<LogicOperations::logicalOr, float, int32_t>;
		break;
		case 33:
			return &filterFunctionConstConst<LogicOperations::logicalOr, float, int64_t>;
		break;
		case 34:
			return &filterFunctionConstConst<LogicOperations::logicalOr, float, float>;
		break;
		case 35:
			return &filterFunctionConstConst<LogicOperations::logicalOr, float, double>;
		break;
		case 39:
			return &filterFunctionConstConst<LogicOperations::logicalOr, float, int8_t>;
		break;
		case 40:
			return &filterFunctionConstCol<LogicOperations::logicalOr, float, int32_t>;
		break;
		case 41:
			return &filterFunctionConstCol<LogicOperations::logicalOr, float, int64_t>;
		break;
		case 42:
			return &filterFunctionConstCol<LogicOperations::logicalOr, float, float>;
		break;
		case 43:
			return &filterFunctionConstCol<LogicOperations::logicalOr, float, double>;
		break;
		case 47:
			return &filterFunctionConstCol<LogicOperations::logicalOr, float, int8_t>;
		break;
		case 48:
			return &filterFunctionConstConst<LogicOperations::logicalOr, double, int32_t>;
		break;
		case 49:
			return &filterFunctionConstConst<LogicOperations::logicalOr, double, int64_t>;
		break;
		case 50:
			return &filterFunctionConstConst<LogicOperations::logicalOr, double, float>;
		break;
		case 51:
			return &filterFunctionConstConst<LogicOperations::logicalOr, double, double>;
		break;
		case 55:
			return &filterFunctionConstConst<LogicOperations::logicalOr, double, int8_t>;
		break;
		case 56:
			return &filterFunctionConstCol<LogicOperations::logicalOr, double, int32_t>;
		break;
		case 57:
			return &filterFunctionConstCol<LogicOperations::logicalOr, double, int64_t>;
		break;
		case 58:
			return &filterFunctionConstCol<LogicOperations::logicalOr, double, float>;
		break;
		case 59:
			return &filterFunctionConstCol<LogicOperations::logicalOr, double, double>;
		break;
		case 63:
			return &filterFunctionConstCol<LogicOperations::logicalOr, double, int8_t>;
		break;
		case 112:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
		case 113:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
		case 114:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int8_t, float>;
		break;
		case 115:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int8_t, double>;
		break;
		case 119:
			return &filterFunctionConstConst<LogicOperations::logicalOr, int8_t, int8_t>;
		break;
		case 120:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
		case 121:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
		case 122:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int8_t, float>;
		break;
		case 123:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int8_t, double>;
		break;
		case 127:
			return &filterFunctionConstCol<LogicOperations::logicalOr, int8_t, int8_t>;
		break;
		case 128:
			return &filterFunctionColConst<LogicOperations::logicalOr, int32_t, int32_t>;
		break;
		case 129:
			return &filterFunctionColConst<LogicOperations::logicalOr, int32_t, int64_t>;
		break;
		case 130:
			return &filterFunctionColConst<LogicOperations::logicalOr, int32_t, float>;
		break;
		case 131:
			return &filterFunctionColConst<LogicOperations::logicalOr, int32_t, double>;
		break;
		case 135:
			return &filterFunctionColConst<LogicOperations::logicalOr, int32_t, int8_t>;
		break;
		case 136:
			return &filterFunctionColCol<LogicOperations::logicalOr, int32_t, int32_t>;
		break;
		case 137:
			return &filterFunctionColCol<LogicOperations::logicalOr, int32_t, int64_t>;
		break;
		case 138:
			return &filterFunctionColCol<LogicOperations::logicalOr, int32_t, float>;
		break;
		case 139:
			return &filterFunctionColCol<LogicOperations::logicalOr, int32_t, double>;
		break;
		case 143:
			return &filterFunctionColCol<LogicOperations::logicalOr, int32_t, int8_t>;
		break;
		case 144:
			return &filterFunctionColConst<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
		case 145:
			return &filterFunctionColConst<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
		case 146:
			return &filterFunctionColConst<LogicOperations::logicalOr, int64_t, float>;
		break;
		case 147:
			return &filterFunctionColConst<LogicOperations::logicalOr, int64_t, double>;
		break;
		case 151:
			return &filterFunctionColConst<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
		case 152:
			return &filterFunctionColCol<LogicOperations::logicalOr, int64_t, int32_t>;
		break;
		case 153:
			return &filterFunctionColCol<LogicOperations::logicalOr, int64_t, int64_t>;
		break;
		case 154:
			return &filterFunctionColCol<LogicOperations::logicalOr, int64_t, float>;
		break;
		case 155:
			return &filterFunctionColCol<LogicOperations::logicalOr, int64_t, double>;
		break;
		case 159:
			return &filterFunctionColCol<LogicOperations::logicalOr, int64_t, int8_t>;
		break;
		case 160:
			return &filterFunctionColConst<LogicOperations::logicalOr, float, int32_t>;
		break;
		case 161:
			return &filterFunctionColConst<LogicOperations::logicalOr, float, int64_t>;
		break;
		case 162:
			return &filterFunctionColConst<LogicOperations::logicalOr, float, float>;
		break;
		case 163:
			return &filterFunctionColConst<LogicOperations::logicalOr, float, double>;
		break;
		case 167:
			return &filterFunctionColConst<LogicOperations::logicalOr, float, int8_t>;
		break;
		case 168:
			return &filterFunctionColCol<LogicOperations::logicalOr, float, int32_t>;
		break;
		case 169:
			return &filterFunctionColCol<LogicOperations::logicalOr, float, int64_t>;
		break;
		case 170:
			return &filterFunctionColCol<LogicOperations::logicalOr, float, float>;
		break;
		case 171:
			return &filterFunctionColCol<LogicOperations::logicalOr, float, double>;
		break;
		case 175:
			return &filterFunctionColCol<LogicOperations::logicalOr, float, int8_t>;
		break;
		case 176:
			return &filterFunctionColConst<LogicOperations::logicalOr, double, int32_t>;
		break;
		case 177:
			return &filterFunctionColConst<LogicOperations::logicalOr, double, int64_t>;
		break;
		case 178:
			return &filterFunctionColConst<LogicOperations::logicalOr, double, float>;
		break;
		case 179:
			return &filterFunctionColConst<LogicOperations::logicalOr, double, double>;
		break;
		case 183:
			return &filterFunctionColConst<LogicOperations::logicalOr, double, int8_t>;
		break;
		case 184:
			return &filterFunctionColCol<LogicOperations::logicalOr, double, int32_t>;
		break;
		case 185:
			return &filterFunctionColCol<LogicOperations::logicalOr, double, int64_t>;
		break;
		case 186:
			return &filterFunctionColCol<LogicOperations::logicalOr, double, float>;
		break;
		case 187:
			return &filterFunctionColCol<LogicOperations::logicalOr, double, double>;
		break;
		case 191:
			return &filterFunctionColCol<LogicOperations::logicalOr, double, int8_t>;
		break;
		case 240:
			return &filterFunctionColConst<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
		case 241:
			return &filterFunctionColConst<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
		case 242:
			return &filterFunctionColConst<LogicOperations::logicalOr, int8_t, float>;
		break;
		case 243:
			return &filterFunctionColConst<LogicOperations::logicalOr, int8_t, double>;
		break;
		case 247:
			return &filterFunctionColConst<LogicOperations::logicalOr, int8_t, int8_t>;
		break;
		case 248:
			return &filterFunctionColCol<LogicOperations::logicalOr, int8_t, int32_t>;
		break;
		case 249:
			return &filterFunctionColCol<LogicOperations::logicalOr, int8_t, int64_t>;
		break;
		case 250:
			return &filterFunctionColCol<LogicOperations::logicalOr, int8_t, float>;
		break;
		case 251:
			return &filterFunctionColCol<LogicOperations::logicalOr, int8_t, double>;
		break;
		case 255:
			return &filterFunctionColCol<LogicOperations::logicalOr, int8_t, int8_t>;
		break;
		default:
			return &invalidArgumentTypeHandler<LogicOperations::logicalOr>;
		break;
	}
}

__device__ DispatchFunction add_gpu_logicalNot_function(int32_t dataTypes)
{
	switch(dataTypes)
	{
		case 0:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 1:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 2:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 3:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 4:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 5:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 6:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 7:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 8:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 9:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 10:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 11:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 12:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 13:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 14:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 15:
			return &filterNotFunctionConst<int32_t>;
		break;
		case 16:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 17:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 18:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 19:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 20:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 21:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 22:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 23:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 24:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 25:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 26:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 27:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 28:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 29:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 30:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 31:
			return &filterNotFunctionConst<int64_t>;
		break;
		case 32:
			return &filterNotFunctionConst<float>;
		break;
		case 33:
			return &filterNotFunctionConst<float>;
		break;
		case 34:
			return &filterNotFunctionConst<float>;
		break;
		case 35:
			return &filterNotFunctionConst<float>;
		break;
		case 36:
			return &filterNotFunctionConst<float>;
		break;
		case 37:
			return &filterNotFunctionConst<float>;
		break;
		case 38:
			return &filterNotFunctionConst<float>;
		break;
		case 39:
			return &filterNotFunctionConst<float>;
		break;
		case 40:
			return &filterNotFunctionConst<float>;
		break;
		case 41:
			return &filterNotFunctionConst<float>;
		break;
		case 42:
			return &filterNotFunctionConst<float>;
		break;
		case 43:
			return &filterNotFunctionConst<float>;
		break;
		case 44:
			return &filterNotFunctionConst<float>;
		break;
		case 45:
			return &filterNotFunctionConst<float>;
		break;
		case 46:
			return &filterNotFunctionConst<float>;
		break;
		case 47:
			return &filterNotFunctionConst<float>;
		break;
		case 48:
			return &filterNotFunctionConst<double>;
		break;
		case 49:
			return &filterNotFunctionConst<double>;
		break;
		case 50:
			return &filterNotFunctionConst<double>;
		break;
		case 51:
			return &filterNotFunctionConst<double>;
		break;
		case 52:
			return &filterNotFunctionConst<double>;
		break;
		case 53:
			return &filterNotFunctionConst<double>;
		break;
		case 54:
			return &filterNotFunctionConst<double>;
		break;
		case 55:
			return &filterNotFunctionConst<double>;
		break;
		case 56:
			return &filterNotFunctionConst<double>;
		break;
		case 57:
			return &filterNotFunctionConst<double>;
		break;
		case 58:
			return &filterNotFunctionConst<double>;
		break;
		case 59:
			return &filterNotFunctionConst<double>;
		break;
		case 60:
			return &filterNotFunctionConst<double>;
		break;
		case 61:
			return &filterNotFunctionConst<double>;
		break;
		case 62:
			return &filterNotFunctionConst<double>;
		break;
		case 63:
			return &filterNotFunctionConst<double>;
		break;
		case 112:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 113:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 114:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 115:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 116:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 117:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 118:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 119:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 120:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 121:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 122:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 123:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 124:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 125:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 126:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 127:
			return &filterNotFunctionConst<int8_t>;
		break;
		case 128:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 129:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 130:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 131:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 132:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 133:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 134:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 135:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 136:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 137:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 138:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 139:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 140:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 141:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 142:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 143:
			return &filterNotFunctionCol<int32_t>;
		break;
		case 144:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 145:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 146:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 147:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 148:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 149:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 150:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 151:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 152:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 153:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 154:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 155:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 156:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 157:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 158:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 159:
			return &filterNotFunctionCol<int64_t>;
		break;
		case 160:
			return &filterNotFunctionCol<float>;
		break;
		case 161:
			return &filterNotFunctionCol<float>;
		break;
		case 162:
			return &filterNotFunctionCol<float>;
		break;
		case 163:
			return &filterNotFunctionCol<float>;
		break;
		case 164:
			return &filterNotFunctionCol<float>;
		break;
		case 165:
			return &filterNotFunctionCol<float>;
		break;
		case 166:
			return &filterNotFunctionCol<float>;
		break;
		case 167:
			return &filterNotFunctionCol<float>;
		break;
		case 168:
			return &filterNotFunctionCol<float>;
		break;
		case 169:
			return &filterNotFunctionCol<float>;
		break;
		case 170:
			return &filterNotFunctionCol<float>;
		break;
		case 171:
			return &filterNotFunctionCol<float>;
		break;
		case 172:
			return &filterNotFunctionCol<float>;
		break;
		case 173:
			return &filterNotFunctionCol<float>;
		break;
		case 174:
			return &filterNotFunctionCol<float>;
		break;
		case 175:
			return &filterNotFunctionCol<float>;
		break;
		case 176:
			return &filterNotFunctionCol<double>;
		break;
		case 177:
			return &filterNotFunctionCol<double>;
		break;
		case 178:
			return &filterNotFunctionCol<double>;
		break;
		case 179:
			return &filterNotFunctionCol<double>;
		break;
		case 180:
			return &filterNotFunctionCol<double>;
		break;
		case 181:
			return &filterNotFunctionCol<double>;
		break;
		case 182:
			return &filterNotFunctionCol<double>;
		break;
		case 183:
			return &filterNotFunctionCol<double>;
		break;
		case 184:
			return &filterNotFunctionCol<double>;
		break;
		case 185:
			return &filterNotFunctionCol<double>;
		break;
		case 186:
			return &filterNotFunctionCol<double>;
		break;
		case 187:
			return &filterNotFunctionCol<double>;
		break;
		case 188:
			return &filterNotFunctionCol<double>;
		break;
		case 189:
			return &filterNotFunctionCol<double>;
		break;
		case 190:
			return &filterNotFunctionCol<double>;
		break;
		case 191:
			return &filterNotFunctionCol<double>;
		break;
		case 240:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 241:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 242:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 243:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 244:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 245:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 246:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 247:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 248:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 249:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 250:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 251:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 252:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 253:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 254:
			return &filterNotFunctionCol<int8_t>;
		break;
		case 255:
			return &filterNotFunctionCol<int8_t>;
		break;
		default:
			return &invalidNotArgumentTypeHandler;
		break;
	}
}



__global__ void fill_gpu_dispatch_table(DispatchFunction* gpuDispatchTable, int32_t gpuDispatchTableSize)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < gpuDispatchTableSize; i += stride)
    {
		int32_t operation = i / OPERATIONS_COUNT;
		int32_t dataTypes = i % (DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE);

		switch (operation)
		{
		case 0:
			gpuDispatchTable[i] = add_gpu_greater_function(dataTypes);
			break;
		case 1:
			gpuDispatchTable[i] = add_gpu_less_function(dataTypes);
			break;
		case 2:
			gpuDispatchTable[i] = add_gpu_greaterEqual_function(dataTypes);
			break;
		case 3:
			gpuDispatchTable[i] = add_gpu_lessEqual_function(dataTypes);
			break;
		case 4:
			gpuDispatchTable[i] = add_gpu_equal_function(dataTypes);
			break;
		case 5:
			gpuDispatchTable[i] = add_gpu_notEqual_function(dataTypes);
			break;
		case 6:
			gpuDispatchTable[i] = add_gpu_logicalAnd_function(dataTypes);
			break;
		case 7:
			gpuDispatchTable[i] = add_gpu_logicalOr_function(dataTypes);
			break;
		case 8:
			gpuDispatchTable[i] = add_gpu_logicalNot_function(dataTypes);
			break;
		default:
			break;
		}
	}
}

__device__ int8_t invalidNotArgumentTypeHandler(GPUOpCode opCode, int32_t offset)
{
	return 2;
}

void FillGpuDispatchTable(DispatchFunction* gpuDispatchTable, int32_t gpuDispatchTableSize)
{
    fill_gpu_dispatch_table<<<1, 512>>>(gpuDispatchTable, gpuDispatchTableSize);
}

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int8_t registers[3];
		for (int32_t j = 0; j < opCodesCount; j++)
		{
            registers[opCodes[i].regIdx] = opCodes[i].fun_ptr(opCodes[i], i);
		}
        outMask[i] = registers[opCodes[opCodesCount - 1].regIdx]; 
    }
}