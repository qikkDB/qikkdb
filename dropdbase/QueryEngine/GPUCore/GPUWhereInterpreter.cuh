#pragma once

#include "GPUWhereInterpreter.h"
#include "MaybeDeref.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

template <typename OP, typename L, typename R>
__device__ void filterFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	R right = gpuStack.pop<R>();
	L left = gpuStack.pop<L>();
	gpuStack.push<int8_t>(OP{}.template operator() <L, R> (left, right));
}

template <typename OP, typename T, typename L, typename R>
__device__ void arithmeticFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	R right = gpuStack.pop<R>();
	L left = gpuStack.pop<L>();
	int32_t errorFlag;
	gpuStack.push<T>(OP{}.template operator() < T, L, R > (left, right, &errorFlag, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
}

template <typename OP>
__device__ void dateFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	int64_t left = gpuStack.pop<int64_t>();
	gpuStack.push<int32_t>(OP{}(left));
}

// TODO: Unary Functions

template <typename OP, typename L>
__device__ void logicalNotFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	L left = gpuStack.pop<L>();
	gpuStack.push<int8_t>(OP{}.template operator() <L> (left));
}

template <typename T>
__device__ void pushConstFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	gpuStack.push<T>(*reinterpret_cast<T*>(opCode.data));
}

template <typename T>
__device__ void pushColFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{
	gpuStack.push<T>(reinterpret_cast<T*>(symbols[opCode.data[0]])[offset]);
}

template <>
__device__ void pushConstFunction<NativeGeoPoint>(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols);

template <>
__device__ void pushConstFunction<GPUMemory::GPUPolygon>(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols);

template <>
__device__ void pushColFunction<GPUMemory::GPUPolygon>(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols);


template <typename OP>
__device__ void invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols)
{

}

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount);

__device__ void invalidContainsArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols);

__device__ void invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, void** symbols);


class GPUWhereDispatcher
{
public:
	static void gpuWhere(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount)
	{
		kernel_filter << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, opCodes, opCodesCount, symbols, dataElementCount);
		cudaDeviceSynchronize();
		CheckCudaError(cudaGetLastError());
	}
};