#ifndef GPU_ARITHMETIC_CUH
#define GPU_ARITHMETIC_CUH

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
/// Kernel PLUS
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V, T min, T max>
__global__ void kernel_plus(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BCol[i] > V{}) && (ACol[i] > (max - BCol[i]))) ||
				((BCol[i] < V{}) && (ACol[i] < (min - BCol[i]))))
			{
				atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
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
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V, T min, T max>
__global__ void kernel_minus(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (((BCol[i] > V{}) && (ACol[i] < (min + BCol[i]))) ||
				((BCol[i] < V{}) && (ACol[i] > (max + BCol[i]))))
			{
				atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
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
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V, T min, T max>
__global__ void kernel_multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for overflow
			if (ACol[i] > U{})
			{
				if (BCol[i] > V{})
				{
					if (ACol[i] > (max / BCol[i]))
					{
						atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
						continue;
					}
				}
				else
				{
					if (BCol[i] < (min / ACol[i]))
					{
						atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
						continue;
					}
				}
			}
			else
			{
				if (BCol[i] > V{})
				{
					if (ACol[i] < (min / BCol[i]))
					{
						atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
						continue;
					}
				}
				else
				{
					if ((ACol[i] != U{}) && (BCol[i] < (max / ACol[i])))
					{
						atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_INTEGER_OVERFLOW_ERROR);
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
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_floor_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR);
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
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR);
			}
			else
			{
				output[i] = ACol[i] / (T)BCol[i]; // convert divisor to type T (should be floating point)
			}
		}
		else
		{
			output[i] = ACol[i] / BCol[i];
		}
	}
}

/// <summary>
/// Operation MODULO kernel
/// </summary>
/// <param name="output">output result data block</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <param name="errorFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount, int32_t* errorFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if at least one of the input operands is float
		if (std::is_floating_point<U>::value || std::is_floating_point<V>::value)
		{
			atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_UNSUPPORTED_DATA_TYPE);
		}
		// if none of the input operands are float
		else
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errorFlag, (int32_t)QueryEngineError::GPU_DIVISION_BY_ZERO_ERROR);
			}
			else
			{
				output[i] = ACol[i] % BCol[i];
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUArithmetic
{
private:
	class ErrorFlagSwapper {
	private:
		int32_t * errorFlagPointer;

	public:
		ErrorFlagSwapper() {
			GPUMemory::allocAndSet(&errorFlagPointer, (int32_t)QueryEngineError::GPU_EXTENSION_SUCCESS, 1);
		}

		~ErrorFlagSwapper() {
			int32_t errorFlag;
			GPUMemory::copyDeviceToHost(&errorFlag, errorFlagPointer, 1);
			GPUMemory::free(errorFlagPointer);

			if (errorFlag != QueryEngineError::GPU_EXTENSION_SUCCESS)
			{
				Context::getInstance().getLastError().setType((QueryEngineError::Type)errorFlag);
			}
			else
			{
				Context::getInstance().getLastError().setCudaError(cudaGetLastError());
			}
		}

		int32_t * getFlagPointer() {
			return errorFlagPointer;
		}
	};

public:
	template<typename T, typename U, typename V>
	static void plus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_plus <T, U, V, std::numeric_limits<T>::min(), std::numeric_limits<T>::max()>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());
		
		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void minus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_minus <T, U, V, std::numeric_limits<T>::min(), std::numeric_limits<T>::max()>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_multiplication <T, U, V, std::numeric_limits<T>::min(), std::numeric_limits<T>::max()>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void floorDivision(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_floor_division << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void division(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_division << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

	template<typename T, typename U, typename V>
	static void modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;

		kernel_modulo << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagSwapper.getFlagPointer());

		cudaDeviceSynchronize();
	}

};

#endif 