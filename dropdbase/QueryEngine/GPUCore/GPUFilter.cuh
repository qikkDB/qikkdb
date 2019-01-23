#ifndef GPU_FILTER_CUH
#define GPU_FILTER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for comparing values from two columns - operator greater than (>)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_gt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] > BCol[i];
	}
}

/// <summary>
/// Kernel for comparing values from two columns - operator less than (<)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_lt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] < BCol[i];
	}
}

/// <summary>
/// Kernel for comparing values from two columns - operator greater than or equals (>=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_gt_eq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] >= BCol[i];
	}
}

/// <summary>
/// Kernel for comparing values from two columns - operator less than or equals (<=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_lt_eq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] <= BCol[i];
	}
}

/// <summary>
/// Kernel for comparing values from two columns - operator equals (==)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_eq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] == BCol[i];
	}
}

/// <summary>
/// Kernel for comparing values from two columns - operator non equals (!=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_non_eq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] != BCol[i];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUFilter
{
public:
	// Operator >
	template<typename T, typename U>
	static void gt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_gt << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <
	template<typename T, typename U>
	static void lt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_lt << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator >=
	template<typename T, typename U>
	static void gtEq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_gt_eq << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <=
	template<typename T, typename U>
	static void ltEq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_lt_eq << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator ==
	template<typename T, typename U>
	static void eq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_eq << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator !=
	template<typename T, typename U>
	static void nonEq(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_non_eq << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif
