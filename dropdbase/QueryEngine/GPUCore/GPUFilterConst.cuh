#ifndef GPU_FILTER_CONST_CUH
#define GPU_FILTER_CONST_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] != BConst;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class GPUFilterConst {
public:
	// Operator >
	template<typename T, typename U>
	void gt(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_gt_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <
	template<typename T, typename U>
	void lt(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_lt_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator >=
	template<typename T, typename U>
	void gtEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_gt_eq_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator <=
	template<typename T, typename U>
	void ltEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_lt_eq_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator ==
	template<typename T, typename U>
	void eq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_eq_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Operator !=
	template<typename T, typename U>
	void nonEq(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount) const {
		kernel_non_eq_const<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif 
