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
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U, typename V>
__global__ void kernel_operator_and(T *outCol, U *ACol, V *BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = ACol[i] && BCol[i];
	}
}

/// <summary>
/// Bitwise OR operation kernel between query result Cols
/// </summary>
/// <param name="outCol">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>void</returns>
template<typename T, typename U, typename V>
__global__ void kernel_operator_or(T *outCol, U *ACol, V *BCol, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for(int32_t i = idx; i < dataElementCount; i += stride)
	{
		outCol[i] = ACol[i] || BCol[i];
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

class GPULogic {
public:
	/// <summary>
	/// Bitwise AND operation between columns
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the left input operands</param>
	/// <param name="BCol">block of the right input operands</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template< typename T, typename U, typename V >
	static void and(T *outCol, U *ACol, V *BCol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_and << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		
		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
	}

	/// <summary>
	/// Bitwise OR operation between columns
	/// </summary>
	/// <param name="outCol">block of the result data</param>
	/// <param name="ACol">block of the left input operands</param>
	/// <param name="BCol">block of the right input operands</param>
	/// <param name="dataElementCount">the size of the input blocks in elements</param>
	/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
	template<typename T, typename U, typename V>
	static void or(T *outCol, U *ACol, V *BCol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_operator_or << <  context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		
		// Get last error
		context.getLastError().setCudaError(cudaGetLastError());
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
};

#endif 
