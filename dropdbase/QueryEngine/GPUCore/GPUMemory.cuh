#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Context.h"
#include "../CudaMemAllocator.h"

// Memory methods
class GPUMemory {
public:
	// Memory allocation
	template<typename T>
	static void alloc(T **p_Block, int32_t dataElementCount)
	{
		*p_Block = reinterpret_cast<T*>(CudaMemAllocator::GetInstance().allocate(dataElementCount * sizeof(T)));
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// malloc + memset
	template<typename T>
	static void allocAndSet(T **p_Block, T value, int32_t dataElementCount)
	{
		*p_Block = reinterpret_cast<T*>(CudaMemAllocator::GetInstance().allocate(dataElementCount * sizeof(T)));

		fill(*p_Block, value, dataElementCount);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Fill an array with a desired value
	template<typename T>
	static void fill(T *p_Block, T value, int32_t dataElementCount)
	{
		//cudaMemsetAsync(p_Block, value, dataElementCount * sizeof(T));	// Async version, uncomment if needed
		cudaMemset(p_Block, value, dataElementCount * sizeof(T));
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Moving data from host to device
	template<typename T>
	static void copyHostToDevice(T *p_BlockDevice, T *p_BlockHost, int32_t dataElementCount)
	{
		cudaMemcpy(p_BlockDevice, p_BlockHost, dataElementCount * sizeof(T), cudaMemcpyHostToDevice);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Moving data from device to host
	template<typename T>
	static void copyDeviceToHost(T *p_BlockHost, T *p_BlockDevice, int32_t dataElementCount)
	{
		cudaMemcpy(p_BlockHost, p_BlockDevice, dataElementCount * sizeof(T), cudaMemcpyDeviceToHost);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Freeing data
	template<typename T>
	static void free(T *p_block)
	{
		CudaMemAllocator::GetInstance().deallocate(reinterpret_cast<int8_t*>(p_block), 0);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename T>
	static void hostRegister(T **devicePtr, T *hostPtr, int32_t dataElementCount)
	{
		cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterMapped);
		cudaHostGetDevicePointer(devicePtr, hostPtr, 0);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename T>
	static void hostUnregister(T *hostPtr)
	{
		cudaHostUnregister(hostPtr);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Pin host memory
	template<typename T>
	static void hostPin(T* hostPtr, int32_t dataElementCount)
	{
		cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterDefault);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Wipe all allocated memory O(1)
	static void clear()
	{
		CudaMemAllocator::GetInstance().Clear();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif