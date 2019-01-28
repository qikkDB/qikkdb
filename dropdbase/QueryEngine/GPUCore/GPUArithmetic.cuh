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

///////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Kernel PLUS
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_plus(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BCol[i] > V{ 0 }) && (ACol[i] > (max - BCol[i]))) ||
				((BCol[i] < V{ 0 }) && (ACol[i] < (min - BCol[i]))))
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
				continue;
			}
		}
		output[i] = ACol[i] + BCol[i];
	}
}

/// <summary>
/// Kernel MINUS
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_minus(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BCol[i] > V{ 0 }) && (ACol[i] < (min + BCol[i]))) ||
				((BCol[i] < V{ 0 }) && (ACol[i] > (max + BCol[i]))))
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
				continue;
			}
		}
		output[i] = ACol[i] - BCol[i];
	}
}

/// <summary>
/// Kernel MULTIPLICATION
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (ACol[i] > U{ 0 })
			{
				if (BCol[i] > V{ 0 })
				{
					if (ACol[i] > (max / BCol[i]))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
				else
				{
					if (BCol[i] < (min / ACol[i]))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
			}
			else
			{
				if (BCol[i] > V{ 0 })
				{
					if (ACol[i] < (min / BCol[i]))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
				else
				{
					if ((ACol[i] != U{ 0 }) && (BCol[i] < (max / ACol[i])))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
			}
		}
		output[i] = ACol[i] * BCol[i];
	}
}

/// <summary>
/// Kernel FLOOR DIVISION
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_floor_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for zero division
			if (BCol[i] == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR));
			}
			else
			{
				output[i] = ACol[i] / BCol[i];
			}
		}
		else
		{
			output[i] = floorf(ACol[i] / BCol[i]);
		}
	}
}

/// <summary>
/// Kernel DIVISION - as T always use some kind of floating point type!
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			output[i] = ACol[i] / static_cast<T>(BCol[i]); // convert divisor to type T (should be floating point)
		}
		else
		{
			output[i] = ACol[i] / BCol[i];
		}
	}
}

/// <summary>
/// Kernel MODULO - as U and V never use floating point type!
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">count of elements in the input blocks</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// Check for zero division
		if (BCol[i] == V{ 0 })
		{
			atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR));
		}
		else
		{
			output[i] = ACol[i] % BCol[i];
		}
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