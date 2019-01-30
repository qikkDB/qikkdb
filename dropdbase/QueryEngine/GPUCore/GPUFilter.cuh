#ifndef GPU_FILTER_CUH
#define GPU_FILTER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"

namespace FilterConditions
{
	struct greater
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a > b;
		}
	};

	struct greaterEquals
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a >= b;
		}
	};

	struct less
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a < b;
		}
	};

	struct lessEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a <= b;
		}
	};

	struct equal
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a == b;
		}
	};

	struct notEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a != b;
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for comparing values from two columns
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename FILTER, typename T, typename U>
__global__ void kernel_filter_col_col(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = FILTER{}(ACol[i], BCol[i]);
	}
}

/// <summary>
/// Kernel for comparing values from column with constant
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">constant to compare</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename FILTER, typename T, typename U>
__global__ void kernel_filter_col_const(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = FILTER{}(ACol[i], BConst);
	}
}

/// <summary>
/// Kernel for comparing constant with values from column
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="AConst">left input operand</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename FILTER, typename T, typename U>
__global__ void kernel_filter_const_col(int8_t *outMask, T AConst, U* BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = FILTER{}(AConst, BCol[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUFilter
{
public:
	template<typename FILTER, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_filter_col_col <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_filter_col_const <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_filter_const_col <FILTER> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename FILTER, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		GPUMemory::fill(outMask, FILTER{}(AConst, BConst), dataElementCount);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif
