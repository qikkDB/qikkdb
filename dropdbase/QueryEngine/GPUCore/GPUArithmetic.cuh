#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "../NullConstants.cuh"
#include "ArithmeticOperations.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic operation with column and column
/// (For mod as U and V never use floating point type!)
/// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename OP, typename T, typename U, typename V>
__global__ void kernel_arithmetic(T* output, U ACol, V BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = OP{}.template operator()
			< T,
			typename std::remove_pointer<U>::type,
			typename std::remove_pointer<V>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i),
				errorFlag,
				min,
				max);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for binary arithmetic functions
class GPUArithmetic
{
public:
	/// Arithmetic operation with two columns
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
	/// <param name="output">output GPU buffer</param>
	/// <param name="ACol">buffer with left side operands</param>
	/// <param name="BCol">buffer with right side operands</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U, typename V>
	static void colCol(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}

	/// Arithmetic operation with column and constant
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
	/// <param name="output">output GPU buffer</param>
	/// <param name="ACol">buffer with left side operands</param>
	/// <param name="BConst">right side operand constant</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U, typename V>
	static void colConst(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}

	/// Arithmetic operation with constant and column
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
	/// <param name="output">output GPU buffer</param>
	/// <param name="AConst">left side operand constant</param>
	/// <param name="BCol">buffer with right side operands</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U, typename V>
	static void constCol(T *output, U AConst, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, BCol, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}

	/// Arithmetic operation with two constants
    /// <param name="OP">Template parameter for the choice of the arithmetic operation</param>
	/// <param name="output">output GPU buffer</param>
	/// <param name="AConst">left side operand constant</param>
	/// <param name="BConst">right side operand constant</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U, typename V>
	static void constConst(T *output, U AConst, V BConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, BConst, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}
};
