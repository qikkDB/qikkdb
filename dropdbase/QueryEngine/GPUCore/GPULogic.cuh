#ifndef GPU_LOGIC_CUH
#define GPU_LOGIC_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace LogicOperations
{
	struct and
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b)
		{
			return a && b;
		}

	};

	struct or
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b)
		{
			return a || b;
		}
	};

}



/// <summary>
/// Bitwise AND operation kernel between query result Cols
/// Requires two int8_t block Cols
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename OP, typename T, typename U, typename V>
__global__ void kernel_logic_col_col(T *outCol, U *ACol, V *BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}(ACol[i], BCol[i]);
	}
}

template<typename OP, typename T, typename U, typename V>
__global__ void kernel_logic_col_const(T *outCol, U *ACol, V BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}(ACol[i], BConst);
	}
}

template<typename OP, typename T, typename U, typename V>
__global__ void kernel_logic_const_col(T *outCol, U AConst, V *BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}(AConst, BCol[i]);
	}
}

template<typename OP, typename T, typename U, typename V>
__global__ void kernel_logic_const_const(T *outCol, U AConst, V BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = OP{}(AConst, BConst);
	}
}

/// <summary>
/// NOT operation kernel on a result Col
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U>
__global__ void kernel_operator_not(T *outCol, U *ACol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = !ACol[i];
	}
}

/// <summary>
/// NOT operation kernel on a const
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="AConst">Const to be negated</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U>
__global__ void kernel_operator_not_const(T *outCol, U AConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = !AConst;
	}
}


class GPULogic {
public:
	template<typename OP, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_logic_col_col <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_logic_col_const <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_logic_const_col <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		kernel_logic_const_const <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BConst, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
	
	
	
	/// <summary>
	/// NOT operation on column
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the input operands</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename T, typename U>
	static void not(T *outCol, U *ACol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_not << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, dataElementCount);
		cudaDeviceSynchronize();
		
		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
	}

	/// <summary>
	/// NOT operation on const
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="AConst">constant to be negated</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename T, typename U>
	static void not_const(T *outCol, U AConst, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_not_const << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, AConst, dataElementCount);
		cudaDeviceSynchronize();

		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
	}
};

#endif 
