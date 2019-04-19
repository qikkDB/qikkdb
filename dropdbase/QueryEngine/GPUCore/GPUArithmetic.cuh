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

namespace ArithmeticOperations
{
	struct add
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (((b > V{ 0 }) && (a > (max - b))) ||
					((b < V{ 0 }) && (a < (min - b))))
				{
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
					return GetNullConstant<T>();
				}
			}
			return a + b;
		}
	};

	struct sub
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (((b > V{ 0 }) && (a < (min + b))) ||
					((b < V{ 0 }) && (a > (max + b))))
				{
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
					return GetNullConstant<T>();
				}
			}
			return a - b;
		}
	};

	struct mul
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (a > U{ 0 })
				{
					if (b > V{ 0 })
					{
						if (a > (max / b))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
					else
					{
						if (b < (min / a))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
				}
				else
				{
					if (b > V{ 0 })
					{
						if (a < (min / b))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
					else
					{
						if ((a != U{ 0 }) && (b < (max / a)))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
				}
			}
			return a * b;
		}
	};

	struct div
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			if (b == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
				return GetNullConstant<T>();
			}
			else
			{
				return a / b;
			}
		}
	};

	struct mod
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			//modulo is not defined for floating point type
			static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
				"None of the input columns of operation modulo cannot be floating point type!");

			// Check for zero division
			if (b == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
				return GetNullConstant<T>();
			}

			return a % b;
		}
	};

	struct bitwiseAnd
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a & b;
		}
	};

	struct bitwiseOr
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a | b;
		}
	};

	struct bitwiseXor
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a ^ b;
		}
	};

	struct bitwiseLeftShift
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a << b;
		}
	};

	struct bitwiseRightShift
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a >> b;
		}
	};

	struct logarithm
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return logf(a) / logf(b);
		}
	};

	struct power
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, b);
		}
	};

	struct root
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, 1/b);
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for arithmetic operation with column and column
/// (For mod as U and V never use floating point type!)
/// </summary>
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

class GPUArithmetic
{
public:
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
