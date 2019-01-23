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
		Context::getInstance().getLastError().setCudaError(cudaSuccess);
	}

	// malloc + memset
	template<typename T>
	static void allocAndSet(T **p_Block, T value, int32_t dataElementCount)
	{
		*p_Block = CudaMemAllocator::GetInstance().allocate(dataElementCount * sizeof(T));

		fill(*p_Block, value, dataElementCount);

		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Fill an array with a desired value
	template<typename T>
	static void fill(T *p_Block, T value, int32_t dataElementCount)
	{
		//cudaError_t cudaStatus = cudaMemsetAsync(p_Block, value, dataElementCount * sizeof(T));	// Async version, uncomment if needed
		cudaError_t cudaStatus = cudaMemset(p_Block, value, dataElementCount * sizeof(T));
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Moving data from host to device
	template<typename T>
	static void copyHostToDevice(T *p_BlockDevice, T *p_BlockHost, int32_t dataElementCount)
	{
		cudaError_t cudaStatus = cudaMemcpy(p_BlockDevice, p_BlockHost, dataElementCount * sizeof(T),
			cudaMemcpyHostToDevice);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Moving data from device to host
	template<typename T>
	static void copyDeviceToHost(T *p_BlockHost, T *p_BlockDevice, int32_t dataElementCount)
	{
		cudaError_t cudaStatus = cudaMemcpy(p_BlockHost, p_BlockDevice, dataElementCount * sizeof(T),
			cudaMemcpyDeviceToHost);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Freeing data
	template<typename T>
	static void free(T *p_block)
	{
		CudaMemAllocator::GetInstance().deallocate(reinterpret_cast<int8_t*>(p_block), 0);
		Context::getInstance().getLastError().setCudaError(cudaSuccess);
	}

	template<typename T>
	static void hostRegister(T **devicePtr, T *hostPtr, int32_t dataElementCount)
	{
		cudaError_t cudaStatus = cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterMapped);
		cudaStatus = cudaHostGetDevicePointer(devicePtr, hostPtr, 0);

		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	template<typename T>
	static void hostUnregister(T *hostPtr)
	{
		cudaError_t cudaStatus = cudaHostUnregister(hostPtr);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Pin host memory
	template<typename T>
	static void hostPin(T* hostPtr, int32_t dataElementCount)
	{
		cudaError_t cudaStatus = cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterDefault);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Wipe all allocated memory O(1)
	static void clear()
	{
		CudaMemAllocator::GetInstance().Clear();
	}

};

#endif