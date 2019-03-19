#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"

namespace LogicOperations
{
	struct logicalAnd
	{
		template<typename U, typename V>
		__device__ __host__ int8_t operator()(U a, V b)
		{
			return a && b;
		}

	};

	struct logicalOr
	{
		template<typename U, typename V>
		__device__ __host__ int8_t operator()(U a, V b)
		{
			return a || b;
		}
	};

	struct logicalNot
	{
		template<typename U>
		__device__ __host__ int8_t operator()(U a)
		{
			return !a;
		}
	};
}

/// <summary>
/// Bitwise AND operation kernel between query result Cols
/// Requires two int8_t block Cols
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename OP, typename U, typename V>
__global__ void kernel_logic(int8_t *outCol, U ACol, V BCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}.template operator()
			<
			typename std::remove_pointer<U>::type,
			typename std::remove_pointer<V>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i));
	}
}

/// <summary>
/// NOT operation kernel on a result Col
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename U>
__global__ void kernel_operator_not(int8_t *outCol, U ACol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = !maybe_deref<typename std::remove_pointer<U>::type>(ACol, i);
	}
}


class GPULogic {
public:
	template<typename OP, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BConst, dataElementCount);
		QueryEngineError::setCudaError(cudaGetLastError());
	}
	
	
	/// <summary>
	/// NOT operation on column
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the input operands</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename U>
	static void not_col(int8_t *outCol, U *ACol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();
		kernel_operator_not << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, dataElementCount);
		
		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	/// <summary>
	/// NOT operation on const
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="AConst">constant to be negated</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename U>
	static void not_const(int8_t *outCol, U AConst, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_not << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, AConst, dataElementCount);

		// Get last error
		QueryEngineError::setCudaError(cudaGetLastError());
	}
};
