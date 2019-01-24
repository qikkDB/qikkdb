#ifndef GPU_FILTER_CONST_CUH
#define GPU_FILTER_CONST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Kernel for comparing values from column with constant - operator greater than (>)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">constant to compare</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_gt_const(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] > BConst;
	}
}

/// <summary>
/// Kernel for comparing values from column with constant - operator less than (<)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_lt_const(int8_t *outMask, T *ACol, U *BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] < BConst;
	}
}

/// <summary>
/// Kernel for comparing values from column with constant - operator greater than or equals (>=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_gt_eq_const(int8_t *outMask, T *ACol, U *BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] >= BConst;
	}
}

/// <summary>
/// Kernel for comparing values from column with constant - operator less than or equals (<=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_lt_eq_const(int8_t *outMask, T *ACol, U *BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] <= BConst;
	}
}

/// <summary>
/// Kernel for comparing values from column with constant - operator equals (==)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_eq_const(int8_t *outMask, T *ACol, U *BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] == BConst;
	}
}

/// <summary>
/// Kernel for comparing values from column with constant - operator non equals (!=)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T, typename U>
__global__ void kernel_non_eq_const(int8_t *outMask, T *ACol, U *BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] != BConst;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUFilterConst
{
public:
	// Operator >
	template<typename T, typename U>
	static void gt(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_gt_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <
	template<typename T, typename U>
	static void lt(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_lt_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator >=
	template<typename T, typename U>
	static void gtEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_gt_eq_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <=
	template<typename T, typename U>
	static void ltEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_lt_eq_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator ==
	template<typename T, typename U>
	static void eq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_eq_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator !=
	template<typename T, typename U>
	static void nonEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_non_eq_const << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif 
