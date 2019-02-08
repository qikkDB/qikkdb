#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"
#include "GPUConstants.cuh"

namespace FilterConditions
{
	struct greater
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a > b;
		}
	};

	struct greaterEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a >= b;
		}
	};

	struct less
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a < b;
		}
	};

	struct lessEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a <= b;
		}
	};

	struct equal
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a == b;
		}
	};

	struct notEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a != b;
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for comparing values 
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename FILTER, typename T, typename U>
__global__ void kernel_filter(int8_t *outMask, T ACol, U BCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;
	const int32_t loopIterations = (dataElementCount + stride - 1 - idx) / stride;
	const int32_t alignedLoopIterations = loopIterations - (loopIterations % UNROLL_FACTOR);
	const int32_t alignedDataElementCount = alignedLoopIterations * stride + idx;

	//unroll from idx to alignedDataElementCount
	#pragma unroll UNROLL_FACTOR
	for (int32_t i = idx; i < alignedDataElementCount; i += stride)
	{
		outMask[i] = FILTER{}.template operator()
			< typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i));
	}
	//continue classic way from alignedDataElementCount to full dataElementCount
	for (int32_t i = alignedDataElementCount; i < dataElementCount; i += stride)
	{
		outMask[i] = FILTER{}.template operator()
			< typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUFilter
{
public:
	template<typename FILTER, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_filter <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_filter <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_filter <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		GPUMemory::memset(outMask, FILTER{}(AConst, BConst), dataElementCount);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

