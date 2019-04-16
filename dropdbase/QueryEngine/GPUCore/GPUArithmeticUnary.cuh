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
#include "../QueryEngineError.h"
#include "MaybeDeref.cuh"

namespace ArithmeticUnaryOperations
{
	struct minus
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return -a;
		}
	};

	struct absolute
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return abs(a);
		}
	};

	struct sine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return sinf(a);
		}
	};

	struct cosine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return cosf(a);
		}
	};

	struct tangent
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return tanf(a);
		}
	};

	struct arcsine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return asinf(a);
		}
	};

	struct arccosine
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return acosf(a);
		}
	};

	struct arctangent
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return atanf(a);
		}
	};

	struct logarithm10
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return log10f(a);
		}
	};

	struct logarithmNatural
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return logf(a);
		}
	};

	struct exponential
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return expf(a);
		}
	};

	struct squareRoot
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return sqrtf(a);
		}
	};

	struct square
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ T operator()(U a, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, 2);
		}
	};

	struct sign
	{
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ T operator()(U val, int32_t* errorFlag, T min, T max) const
		{
			return (U{ 0 } < val) - (val < U{ 0 });
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for arithmetic unary operation with column and column
/// </summary>
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

class GPUArithmeticUnary
{
public:
	template<typename OP, typename T, typename U>
	static void col(T *output, U *ACol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_arithmetic_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
	}

	template<typename OP, typename T, typename U>
	static void cnst(T *output, U AConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_arithmetic_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
	}
};
