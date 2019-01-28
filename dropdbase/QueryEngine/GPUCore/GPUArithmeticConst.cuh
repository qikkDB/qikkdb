#ifndef GPU_ARITHMETIC_CONST_CUH
#define GPU_ARITHMETIC_CONST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../Context.h"
#include "../QueryEngineError.h"
#include "GPUMemory.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Kernel PLUS constant
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">right input operand</param>
/// <param name="dataElementCount">the size of the input block</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_plus_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag, T minPossible, T maxPossible)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BConst > V{ 0 }) && (ACol[i] > maxPossible)) ||
				((BConst < V{ 0 }) && (ACol[i] < minPossible)))
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
				continue;
			}
		}
		output[i] = ACol[i] + BConst;
	}
}

/// <summary>
/// Kernel MINUS constant
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_minus_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag, T minPossible, T maxPossible)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BConst > V{ 0 }) && (ACol[i] < minPossible)) ||
				((BConst < V{ 0 }) && (ACol[i] > maxPossible)))
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
				continue;
			}
		}
		output[i] = ACol[i] - BConst;
	}
}

/// <summary>
/// Kernel MULTIPLICATION constant
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_multiplication_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag, T min, T max)
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
				if (BConst > V{ 0 })
				{
					if (ACol[i] > (max / BConst))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
				else
				{
					if (BConst < (min / ACol[i]))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
			}
			else
			{
				if (BConst > V{ 0 })
				{
					if (ACol[i] < (min / BConst))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
				else
				{
					if ((ACol[i] != U{ 0 }) && (BConst < (max / ACol[i])))
					{
						atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR));
						continue;
					}
				}
			}
		}
		output[i] = ACol[i] * BConst;
	}
}

/// <summary>
/// Kernel FLOOR DIVISION constant
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_floor_division_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for zero division
			if (BConst == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR));
			}
			else
			{
				output[i] = ACol[i] / BConst;
			}
		}
		else
		{
			output[i] = floorf(ACol[i] / BConst);
		}
	}
}

/// <summary>
/// Kernel DIVISION constant - as T always use some kind of floating point type!
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_division_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			output[i] = ACol[i] / static_cast<T>(BConst); // convert divisor to type T (should be floating point)
		}
		else
		{
			output[i] = ACol[i] / BConst;
		}
	}
}

/// <summary>
/// Kernel MODULO constant - as U and V never use floating point type!
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_modulo_const(T *output, U *ACol, V BConst, int32_t dataElementCount, int32_t* errorFlag)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = ACol[i] % BConst;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////



class GPUArithmeticConst {
public:
	template<typename T, typename U, typename V>
	static void plus(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_plus_const <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min() - static_cast<T>(BConst), std::numeric_limits<T>::max() - static_cast<T>(BConst));

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void minus(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_minus_const <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min() + static_cast<T>(BConst), std::numeric_limits<T>::max() + static_cast<T>(BConst));

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void multiplication(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_multiplication_const <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer(),
				std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void floorDivision(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		// Check for zero division
		if (BConst == V{ 0 })
		{
			Context::getInstance().getLastError().setType(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR);
		}
		else
		{
			ErrorFlagSwapper errorFlagSwapper;

			kernel_floor_division_const <T, U, V>
				<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
				(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer());

			cudaDeviceSynchronize();
		}
	}

	template<typename T, typename U, typename V>
	static void division(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		// TODO Uncomment when dispatcher is ready for this
		////result of this type of division operation is always floating point - so check type T
		//static_assert(std::is_floating_point<T>::value,
		//	"Output column of operation division has to be floating point type! For integer division use operation floorDivision.");

		ErrorFlagSwapper errorFlagSwapper;

		kernel_division_const <T, U, V>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void modulo(T *output, U *ACol, V BConst, int32_t dataElementCount)
	{
		//modulo is not defined for floating point type
		static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
			"None of the input columns of operation modulo cannot be floating point type!");

		// Check for zero division
		if (BConst == V{ 0 })
		{
			Context::getInstance().getLastError().setType(QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR);
		}
		else
		{
			ErrorFlagSwapper errorFlagSwapper;

			kernel_modulo_const <T, U, V>
				<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
				(output, ACol, BConst, dataElementCount, errorFlagSwapper.getFlagPointer());

			cudaDeviceSynchronize();
		}
	}
};

#endif 
