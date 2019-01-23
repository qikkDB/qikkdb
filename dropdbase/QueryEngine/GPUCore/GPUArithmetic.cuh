#ifndef GPU_ARITHMETIC_CUH
#define GPU_ARITHMETIC_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_plus(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U> && !std::is_floating_point<V>)
		{
			// Chech for overflow
			if (((BCol[i] > V{}) && (ACol[i] > (std::numeric_limits<T>::max() - BCol[i]))) ||
				((BCol[i] < V{}) && (ACol[i] < (std::numeric_limits<T>::min() - BCol[i]))))
			{
				atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
				return;
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_minus(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U> && !std::is_floating_point<V>)
		{
			// Chech for overflow
			if (((BCol[i] > V{}) && (ACol[i] < (std::numeric_limits<T>::min() + BCol[i]))) ||
				((BCol[i] < V{}) && (ACol[i] > (std::numeric_limits<T>::max() + BCol[i]))))
			{
				atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
				return;
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U> && !std::is_floating_point<V>)
		{
			// Check for overflow
			if (ACol[i] > U{})
			{
				if (BCol[i] > V{})
				{
					if (ACol[i] > (std::numeric_limits<T>::max() / BCol[i]))
					{
						atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
						return;
					}
				}
				else
				{
					if (BCol[i] < (std::numeric_limits<T>::min() / ACol[i]))
					{
						atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
						return;
					}
				}
			}
			else
			{
				if (BCol[i] > V{})
				{
					if (ACol[i] < (std::numeric_limits<T>::min() / BCol[i]))
					{
						atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
						return;
					}
				}
				else
				{
					if ((ACol[i] != U{}) && (BCol[i] < (std::numeric_limits<T>::max() / ACol[i])))
					{
						atomicExch(errFlag, GPU_INTEGER_OVERFLOW_ERROR);
						return;
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_floor_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U> && !std::is_floating_point<V>)
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errFlag, GPU_DIVISION_BY_ZERO_ERROR);
				return;
			}
			output[i] = ACol[i] / BCol[i];
		}
		else
		{
			output[i] = floor(ACol[i] / BCol[i]);
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_division(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if none of the input operands are float
		if (!std::is_floating_point<U> && !std::is_floating_point<V>)
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errFlag, GPU_DIVISION_BY_ZERO_ERROR);
				return;
			}
			output[i] = ACol[i] / (T)BCol[i]; // convert divisor to type T (should be floating point)
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
/// <param name="errFlag">flag for error checking</param>
template<typename T, typename U, typename V>
__global__ void kernel_modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount, QueryEngineError::Type* errFlag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		// if at least one of the input operands is float
		if (std::is_floating_point<U> || std::is_floating_point<V>)
		{
			atomicExch(errFlag, GPU_UNSUPPORTED_DATA_TYPE);
			return;
		}
		// if none of the input operands are float
		else
		{
			// Check for zero division
			if (BCol[i] == V{})
			{
				atomicExch(errFlag, GPU_DIVISION_BY_ZERO_ERROR);
				return;
			}
			output[i] = ACol[i] % BCol[i];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUArithmetic
{
private:
	// Malloc a new error flag and set to success
	static QueryEngineError::Type * preKernel()
	{
		QueryEngineError::Type *errorFlagPointer = nullptr;
		GPUMemory::allocAndSet(&errorFlagPointer, QueryEngineError::GPU_EXTENSION_SUCCESS, 1);
		return errorFlagPointer;
	}

	// Read the error flag and set Context last error
	static void postKernel(QueryEngineError::Type *errorFlagPointer)
	{
		QueryEngineError::Type errFlag;
		GPUMemory::copyDeviceToHost(&errFlag, errorFlagPointer, 1);
		GPUMemory::free(errorFlagPointer);

		if (errFlag != QueryEngineError::GPU_EXTENSION_SUCCESS)
		{
			Context::getInstance().getLastError().setType(errFlag);
		}
		else
		{
			Context::getInstance().getLastError().setCudaError(cudaGetLastError());
		}
	}

public:
	template<typename T, typename U, typename V>
	static void plus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_plus << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

	template<typename T, typename U, typename V>
	static void minus(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_minus << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

	template<typename T, typename U, typename V>
	static void multiplication(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_multiplication << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

	template<typename T, typename U, typename V>
	static void floorDivision(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_floor_division << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

	template<typename T, typename U, typename V>
	static void division(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_division << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

	template<typename T, typename U, typename V>
	static void modulo(T *output, U *ACol, V *BCol, int32_t dataElementCount)
	{
		QueryEngineError::Type* errorFlagPointer = preKernel();

		kernel_modulo << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount, errorFlagPointer);

		cudaDeviceSynchronize();

		postKernel(errorFlagPointer);
	}

};

#endif 