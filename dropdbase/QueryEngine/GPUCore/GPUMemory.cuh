#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Context.cuh"

class GPUMemory {
public:
	// Memory methods

	// Memory allocation
	template<typename T>
	void alloc(T **p_Block, int32_t dataElementCount) const
	{
		cudaError_t cudaStatus = cudaMalloc(p_Block, dataElementCount * sizeof(T));
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// malloc + memset
	template<typename T>
	void allocAndSet(T **p_Block, int32_t size, int32_t dataElementCount, T value) const
	{
		cudaError_t cudaStatus = cudaCalloc(p_Block, dataElementCount * sizeof(T));
		cudaStatus = cudaMemset(ptr, value, dataElementCount * sizeof(T));
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Moving data from host to device
	template<typename T>
	void copyHostToDevice(T *p_BlockDevice, void *p_BlockHost, int32_t dataElementCount) const
	{
		cudaError_t cudaStatus = cudaMemcpy(p_BlockDevice, p_BlockHost, dataElementCount * sizeof(T), 
			cudaMemcpyHostToDevice);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Moving data from device to host
	template<typename T>
	void copyDeviceToHost(T *p_BlockHost, void *p_BlockDevice, int32_t dataElementCount) const
	{
		cudaError_t cudaStatus = cudaMemcpy(p_BlockHost, p_BlockDevice, dataElementCount * sizeof(T), 
			cudaMemcpyDeviceToHost);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	// Freeing data
	template<typename T>
	void free(T *p_block) const
	{
		cudaError_t cudaStatus = cudaFree(p_block);
		Context::getInstance().getLastError().setCudaError(cudaStatus);
	}

	template<typename T>
	void hostRegister(T **devicePtr, void *hostPtr, int32_t dataElementCount) const;

	template<typename T>
	void hostUnregister(T *hostPtr) const;

};

#endif