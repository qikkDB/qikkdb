#pragma once

#include "GPUDispatch.h"
#include "MaybeDeref.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

template <typename OP, typename L, typename R>
__device__ void filterFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	R right = gpuStack.pop<R>();
	L left = gpuStack.pop<L>();

	registers[opCode.data[0]] = OP{}.template operator() <L, R> (left, right);
}

template <typename OP, typename T, typename L, typename R>
__device__ void arithmeticFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	R right = gpuStack.pop<R>();
	L left = gpuStack.pop<L>();
	int32_t errorFlag;
	gpuStack.push<T>(OP{}.template operator() < T, L, R > (left, right, &errorFlag, std::numeric_limits<T>::min(), std::numeric_limits<T>::max()));
}

template <typename OP>
__device__ void logicFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	registers[opCode.data[0]] = OP{}.template operator() < int8_t, int8_t > (registers[opCode.data[0]], registers[opCode.data[1]]);
}

template <typename T>
__device__ void pushConst(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	gpuStack.push<T>(*reinterpret_cast<T*>(opCode.data));
}

template <typename T>
__device__ void pushCol(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	gpuStack.push<T>(reinterpret_cast<T*>(symbols[opCode.data])[offset]);
}

__device__ void containsFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols);

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount);