#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../../cub/cub.cuh"

#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "../CudaMemAllocator.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"


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
		// Minimum reduction - ge tthe buffer size
		void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;
		cub::DeviceReduce::Min(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

		// Allocate temporary storage
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

		// Run minimum reduction - data stays on gpu
		cub::DeviceReduce::Min(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
		GPUMemory::free(tempBuffer);

		cudaDeviceSynchronize();

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
		// Minimum reduction - ge tthe buffer size
		void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;
		cub::DeviceReduce::Max(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

		// Allocate temporary storage
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

		// Run minimum reduction - data stays on gpu
		cub::DeviceReduce::Max(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
		GPUMemory::free(tempBuffer);

		cudaDeviceSynchronize();

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
		// Minimum reduction - ge tthe buffer size
		void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;
		cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

		// Allocate temporary storage
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

		// Run minimum reduction - data stays on gpu
		cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
		GPUMemory::free(tempBuffer);

		cudaDeviceSynchronize();

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
		// Minimum reduction - ge tthe buffer size
		void* tempBuffer = nullptr;
		size_t tempBufferSize = 0;
		cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);

		// Allocate temporary storage
		GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&tempBuffer), tempBufferSize);

		// Run minimum reduction - data stays on gpu
		cub::DeviceReduce::Sum(tempBuffer, tempBufferSize, ACol, outValue, dataElementCount);
		GPUMemory::free(tempBuffer);

		cudaDeviceSynchronize();

		// Divide the result - calculate the average
		GPUArithmetic::colConst<ArithmeticOperations::div, T, T, float>(outValue, outValue, static_cast<float>(dataElementCount), 1);

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
	static void cnt(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// TODO, make this function more useful
		T temp = dataElementCount;
		GPUMemory::copyHostToDevice(outValue, &temp, 1);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
};

