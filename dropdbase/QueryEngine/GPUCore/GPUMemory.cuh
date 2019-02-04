#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "../CudaMemAllocator.h"
#include "../../NativeGeoPoint.h"

template<typename T>
__global__ void kernel_fill_array(T *p_Block, T value, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		p_Block[i] = value;
	}
}

// Memory methods
class GPUMemory {
public:

	struct GPUPolygon
	{
		// Points of polygons
		NativeGeoPoint* polyPoints;
		// Start indexes of each polygon in point array
		int32_t* pointIdx;
		// Number of points of each polygon
		int32_t* pointCount;
		// Start indexes of each complex polygon in polygon array
		int32_t* polyIdx;
		// Number of polygons of each complex polygon
		int32_t* polyCount;
	};

	// Memory allocation
	/// <summary>
	/// Memory allocation of block on the GPU with the respective size of the input parameter type
	/// </summary>
	/// <param name="p_Block">pointer to pointer wich will points to allocated memory block on the GPU</param>
	/// <param name="dataType">type of the resulting buffer</param>
	/// <param name="size">count of elements in the block</param>
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	template<typename T>
	static void alloc(T **p_Block, int32_t dataElementCount)
	{
		*p_Block = reinterpret_cast<T*>(CudaMemAllocator::GetInstance().allocate(dataElementCount * sizeof(T)));
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// malloc + memset
	/// <summary>
	/// Memory allocation of block on the GPU with the respective size of the input parameter type
	/// </summary>
	/// <param name="p_Block">pointer to pointer wich will points to allocated memory block on the GPU</param>
	/// <param name="dataType">type of the resulting buffer</param>
	/// <param name="size">count of elements in the block</param>
	/// <param name="value">value to set the memory to
	/// (always has to be int, because of cudaMemset; and just lowest byte will be used
	/// and all bytes in the allocated buffer will be set to that byte value)</param>
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	template<typename T>
	static void allocAndSet(T **p_Block, int value, int32_t dataElementCount)
	{
		*p_Block = reinterpret_cast<T*>(CudaMemAllocator::GetInstance().allocate(dataElementCount * sizeof(T)));

		memset(*p_Block, value, dataElementCount);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Fill an array with a desired value
	template<typename T>
	static void memset(T *p_Block, int value, int32_t dataElementCount)
	{
		//cudaMemsetAsync(p_Block, value, dataElementCount * sizeof(T));	// Async version, uncomment if needed
		cudaMemset(p_Block, value, dataElementCount * sizeof(T));
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	template<typename T>
	static void fillArray(T *p_Block, T value, int32_t dataElementCount)
	{
		kernel_fill_array << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(p_Block, value, dataElementCount);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Moving data from host to device
		/// <summary>
	/// Copy memory block with dataType numbers from host (RAM, CPU's memory) to device (GPU's memory).
	/// </summary>
	/// <param name="p_BlockDevice">pointer to memory block on device</param>
	/// <param name="p_BlockHost">pointer to memory block on host</param>
	/// <param name="dataType">type of the elements buffer</param>
	/// <param name="size">count of int8_t numbers</param>
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	template<typename T>
	static void copyHostToDevice(T *p_BlockDevice, T *p_BlockHost, int32_t dataElementCount)
	{
		cudaMemcpy(p_BlockDevice, p_BlockHost, dataElementCount * sizeof(T), cudaMemcpyHostToDevice);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Moving data from device to host
		/// <summary>
	/// Copy memory block with dataType numbers from device (GPU's memory) to host (RAM, CPU's memory).
	/// </summary>
	/// <param name="p_BlockHost">pointer to memory block on host</param>
	/// <param name="p_BlockDevice">pointer to memory block on device</param>
	/// <param name="dataType">type of the elements buffer</param>
	/// <param name="size">count of int8_t numbers</param>
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	template<typename T>
	static void copyDeviceToHost(T *p_BlockHost, T *p_BlockDevice, int32_t dataElementCount)
	{
		cudaMemcpy(p_BlockHost, p_BlockDevice, dataElementCount * sizeof(T), cudaMemcpyDeviceToHost);
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

	// Freeing data
		/// <summary>
	/// Free memory block from GPU's memory
	/// </summary>
	/// <param name="p_Block">pointer to memory block (on GPU memory)</param>
	/// <returns>return code tells if operation was successful (GPU_EXTENSION_SUCCESS)
	/// or some error occured (GPU_EXTENSION_ERROR)</returns>
	static void free(void *p_block)
	{
		CudaMemAllocator::GetInstance().deallocate(static_cast<int8_t*>(p_block), 0);
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