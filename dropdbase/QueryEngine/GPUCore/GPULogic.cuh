#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"

/// Logic relation operation functors
namespace LogicOperations
{
	/// A logical binary AND operation
	struct logicalAnd
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a && b;
		}

	};

	/// A logical binary OR operation
	struct logicalOr
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a || b;
		}
	};
}

/// A bitwise relation logic operation kernel
/// <param name="OP">Template parameter for the choice of the logic relation operation</param>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename OP, typename T, typename U, typename V>
__global__ void kernel_logic(T *outCol, U ACol, V BCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}.template operator()
			<
			T,
			typename std::remove_pointer<U>::type,
			typename std::remove_pointer<V>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i));
	}
}

/// An unary NOT operation kernel
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U>
__global__ void kernel_operator_not(T *outCol, U ACol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = !maybe_deref<typename std::remove_pointer<U>::type>(ACol, i);
	}
}

/// A class for relational logic operations
class GPULogic {
public:
    /// Relational binary logic operation with two columns of numerical values
    /// <param name="OP">Template parameter for the choice of the logic operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Relational binary logic operation between a column and a constant
	/// <param name="OP">Template parameter for the choice of the logic operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BConst">right side constant operand</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}


	/// Relational binary logic operation between a column and a constant
    /// <param name="OP">Template parameter for the choice of the logic operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="AConst">left side constant operand</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Relational binary logic operation between two  constants
    /// <param name="OP">Template parameter for the choice of the logic operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="AConst">left side constant operand</param>
    /// <param name="BConst">right side constant operand</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		kernel_logic <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BConst, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}
	
	
	/// Unary NOT operation on a column of data
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the input operands</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	template<typename T, typename U>
	static void not_col(T *outCol, U *ACol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();
		kernel_operator_not << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, dataElementCount);
		
		// Get last error
		CheckCudaError(cudaGetLastError());
	}

	/// Unary NOT operation on a constant value
	/// <param name="outCol">block of the result data</param>
	/// <param name="AConst">constant to be negated</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	template<typename T, typename U>
	static void not_const(T *outCol, U AConst, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_not << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, AConst, dataElementCount);

		// Get last error
		CheckCudaError(cudaGetLastError());
	}
};
