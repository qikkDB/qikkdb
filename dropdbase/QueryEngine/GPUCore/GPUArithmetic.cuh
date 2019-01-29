#ifndef GPU_ARITHMETIC_CUH
#define GPU_ARITHMETIC_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../QueryEngineError.h"


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
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
					continue;
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
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
					continue;
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
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
							continue;
						}
					}
					else
					{
						if (b < (min / a))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
							continue;
						}
					}
				}
				else
				{
					if (b > V{ 0 })
					{
						if (a < (min / b))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
							continue;
						}
					}
					else
					{
						if ((a != U{ 0 }) && (b < (max / a)))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
							continue;
						}
					}
				}
			}
			return a * b;
		}
	};

	struct floorDiv
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for zero division
				if (b == V{ 0 })
				{
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR));
				}
				else
				{
					return = a / b;
				}
			}
			else
			{
				return = floorf(a / b);
			}
		}
	};

	struct div
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				return a / static_cast<T>(b); // convert divisor to type T (should be floating point)
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
			// Check for zero division
			if (b == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR));
			}
			else
			{
				return a % b;
			}
			return T{ 0 };
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Kernel for arithmetic operation with column and column
/// (For div as T always use some kind of floating point type!)
/// (For mod as U and V never use floating point type!)
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename OP, typename T, typename U, typename V>
__global__ void kernel_arithmetic_col_col(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = OP{}(ACol[i], BCol[i], errorFlag, min, max);
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUArithmetic
{
public:
	template<typename T, typename U, typename V>
	static void plus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_plus <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void minus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_minus <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_multiplication <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void floorDivision(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_floor_division <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void division(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		// TODO Uncomment when dispatcher is ready for this
		////result of this type of division operation is always floating point - so check type T
		//static_assert(std::is_floating_point<T>::value,
		//	"Output column of operation division has to be floating point type! For integer division use operation floorDivision.");

		ErrorFlagSwapper errorFlagSwapper;

		kernel_division <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		//modulo is not defined for floating point type
		static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
			"None of the input columns of operation modulo cannot be floating point type!");

		ErrorFlagSwapper errorFlagSwapper;

		kernel_modulo <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

};

#endif 