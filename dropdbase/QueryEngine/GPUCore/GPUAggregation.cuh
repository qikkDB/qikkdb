#ifndef GPU_AGGREGATION_CUH
#define GPU_AGGREGATION_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "GPUMemory.cuh"

class GPUAggregation {
public:
	// Aggregation functions on the whole collumn
	/// <summary>
	/// Find the smallest element in the collumn
	/// </summary>
	/// <param name="outValue">the smallest element- kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename T>
	static void min(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// Malloc a new buffer for the output value
		T *outValueGPUPointer = nullptr;
		GPUMemory::alloc(&outValueGPUPointer, 1);

		// Kernel call
		outValueGPUPointer = thrust::min_element(thrust::device, ACol, ACol + dataElementCount);
		cudaDeviceSynchronize();

		// Copy the generated output to outValue (still in GPU)
		cudaMemcpy(outValue, outValueGPUPointer, sizeof(T), cudaMemcpyDeviceToDevice);

		// Free the memory
		GPUMemory::free(outValueGPUPointer);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
	
	/// <summary>
	/// Find the largest element in the collumn
	/// </summary>
	/// <param name="outValue">the smallest element - kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename T>
	static void max(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// Malloc a new buffer for the output value
		T *outValueGPUPointer = nullptr;
		GPUMemory::alloc(&outValueGPUPointer, 1);

		// Kernel calls here
		outValueGPUPointer = thrust::max_element(thrust::device, ACol, ACol + dataElementCount);
		cudaDeviceSynchronize();

		// Copy the generated output to outValue (still in GPU)
		cudaMemcpy(outValue, outValueGPUPointer, sizeof(T), cudaMemcpyDeviceToDevice);

		// Free the memory
		GPUMemory::free(outValueGPUPointer);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
	
	/// <summary>
	/// Return the sum of elements in the collumn
	/// </summary>
	/// <param name="outValue">the sum of elements - kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename T>
	static void sum(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// Kernel calls here
		T outValueHost = thrust::reduce(thrust::device, ACol, ACol + dataElementCount, (T) 0, thrust::plus<T>());
		cudaDeviceSynchronize();

		// Copy the generated output to outValue (still in GPU)
		GPUMemory::copyHostToDevice(outValue, &outValueHost, 1);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	/// <summary>
	/// Return the average of elements in the collumn as a floating point number
	/// </summary>
	/// <param name="outValue">the average of elements - kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename T>
	static void avg(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// Calculate the sum of all elements
		T outValueHost = thrust::reduce(thrust::device, ACol, ACol + dataElementCount, (T)0, thrust::plus<T>());
		outValueHost /= dataElementCount;
		cudaDeviceSynchronize();

		// Copy the generated output to outValue (still in GPU)
		GPUMemory::copyHostToDevice(outValue, &outValueHost, 1);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
	
	/// <summary>
	/// Return the number of elements in the collumn
	/// </summary>
	/// <param name="outValue">the count of elements - kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename T>
	static void cnt(T *outValue,T *ACol, int32_t dataElementCount)
	{
		// TODO, make this function more useful
		T temp = dataElementCount;
		GPUMemory::copyHostToDevice(outValue, &temp, 1);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
};

#endif 

