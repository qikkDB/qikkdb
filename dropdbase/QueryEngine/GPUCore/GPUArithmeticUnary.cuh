#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"

/// Namespace for unary arithmetic operation generic functors
namespace ArithmeticUnaryOperations
{
	/// Arithmetic unary minus
	struct minus
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return -a;
		}
	};

	/// Arithmetic unary absolute
	struct absolute
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return abs(a);
		}
	};

	/// Mathematical function sine
	struct sine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return sinf(a);
		}
	};

	/// Mathematical function cosine
	struct cosine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return cosf(a);
		}
	};

	/// Mathematical function tangent
	struct tangent
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return tanf(a);
		}
	};

	/// Mathematical function cotangent
	struct cotangent
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return 1.0f/tanf(a);
		}
	};

	/// Mathematical function arcus sine
	struct arcsine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return asinf(a);
		}
	};

	/// Mathematical function arcus cosine
	struct arccosine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return acosf(a);
		}
	};

	/// Mathematical function arcus tangent
	struct arctangent
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return atanf(a);
		}
	};

	/// Mathematical function logarithm with base 10
	struct logarithm10
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return log10f(a);
		}
	};

	/// Mathematical function logarithm with base e
	struct logarithmNatural
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return logf(a);
		}
	};

	/// Mathematical function exponential
	struct exponential
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return expf(a);
		}
	};

	/// Mathematical function square root
	struct squareRoot
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return sqrtf(a);
		}
	};

	/// Mathematical function square
	struct square
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, 2);
		}
	};

	/// Mathematical function sign
	struct sign
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U val, int32_t* errorFlag, T min, T max) const
		{
			return (U{ 0 } < val) - (val < U{ 0 });
		}
	};

	/// Mathematical function round
	struct round
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U val, int32_t* errorFlag, T min, T max) const
		{
			return roundf(val);
		}
	};

	/// Mathematical function floor
	struct floor
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U val, int32_t* errorFlag, T min, T max) const
		{
			return floorf(val);
		}
	};

	/// Mathematical function ceil
	struct ceil
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U val, int32_t* errorFlag, T min, T max) const
		{
			return ceilf(val);
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for arithmetic unary operation with column and column
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename OP, typename T, typename U>
__global__ void kernel_arithmetic_unary(T* output, U ACol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = OP{}.template operator()
			< T,
			typename std::remove_pointer<U>::type>
			(maybe_deref(ACol, i),
				errorFlag,
				min,
				max);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for unary arithmetic functions
class GPUArithmeticUnary
{
public:
	/// Arithmetic unary operation with values from column
	/// <param name="output">output GPU buffer</param>
	/// <param name="ACol">buffer with operands</param>
	/// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void col(T *output, U *ACol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}

	/// Arithmetic unary operation with constant
	/// <param name="output">output GPU buffer</param>
	/// <param name="AConst">operand (constant)</param>
	/// <param name="dataElementCount">data element count of the output buffer (how many times copy result)</param>
	template<typename OP, typename T, typename U>
	static void cnst(T *output, U AConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_arithmetic_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, dataElementCount, errorFlagSwapper.GetFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		errorFlagSwapper.Swap();
	}
};
