#ifndef GPU_LOGIC_CUH
#define GPU_LOGIC_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/// <summary>
/// Bitwise AND operation kernel between query result Cols
/// Requires two int8_t block Cols
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">constant as operand</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U, typename V>
__global__ void kernel_operator_and_const(T *outCol, U *ACol, V BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = ACol[i] && BConst;
	}
}

/// <summary>
/// Bitwise OR operation kernel between query result Cols
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BConst">constant as operand</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U, typename V>
__global__ void kernel_operator_or_const(T *outCol, U *ACol, V BConst, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = ACol[i] || BConst;
	}
}

class GPULogicConst {
public:
	/// <summary>
	/// Bitwise AND operation between column and constant
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the left input operands</param>
	/// <param name="BConst">constant as operand</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template< typename T, typename U, typename V >
	static void and(T *outCol, U *ACol, V BConst, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_and_const << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();

		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
	}

	/// <summary>
	/// Bitwise OR operation between column and constant
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the left input operands</param>
	/// <param name="BConst">constant as operand</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename T, typename U, typename V>
	static void or (T *outCol, U *ACol, V BConst, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_or_const << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, BConst, dataElementCount);
		cudaDeviceSynchronize();

		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
	}
};

#endif 
