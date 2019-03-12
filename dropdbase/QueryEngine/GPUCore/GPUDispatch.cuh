#pragma once

#include "GPUDispatch.h"
#include "MaybeDeref.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

template <typename OP, typename L, typename R>
__device__ int8_t filterFunctionColConst(GPUOpCode opCode, int32_t offset)
{
    L* left = reinterpret_cast<L*>(opCode.dataLeft);
    R right = *reinterpret_cast<R*>(&opCode.dataRight);
    return OP{}.template operator()<typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type>(
        maybe_deref(left, offset), maybe_deref(right, offset));
}

template <typename OP, typename L, typename R>
__device__ int8_t filterFunctionConstCol(GPUOpCode opCode, int32_t offset)
{
	L left = *reinterpret_cast<L*>(&opCode.dataLeft);
	R* right = reinterpret_cast<R*>(opCode.dataRight);
	return OP{}.template operator() < typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type > (
		maybe_deref(left, offset), maybe_deref(right, offset));
}

template <typename OP, typename L, typename R>
__device__ int8_t filterFunctionColCol(GPUOpCode opCode, int32_t offset)
{
	L* left = reinterpret_cast<L*>(opCode.dataLeft);
	R* right = reinterpret_cast<R*>(opCode.dataRight);
	return OP{}.template operator() < typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type > (
		maybe_deref(left, offset), maybe_deref(right, offset));
}

template <typename OP, typename L, typename R>
__device__ int8_t filterFunctionConstConst(GPUOpCode opCode, int32_t offset)
{
	L left = *reinterpret_cast<L*>(&opCode.dataLeft);
	R right = *reinterpret_cast<R*>(&opCode.dataRight);
	return OP{}.template operator() < typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type > (
		maybe_deref(left, offset), maybe_deref(right, offset));
}


template <typename OP>
__device__ int8_t invalidArgumentTypeHandler(GPUOpCode opCode, int32_t offset)
{
	return 2;
}

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, int32_t dataElementCount);

__global__ void fill_gpu_dispatch_table(DispatchFunction* gpuDispatchTable, int32_t gpuDispatchTableSize);