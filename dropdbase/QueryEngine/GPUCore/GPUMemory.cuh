#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

#include "../Context.h"
#include "../CudaMemAllocator.h"
#include "../../NativeGeoPoint.h"

/// Kernel for filling a buffer in parallel with data
/// <param name="p_Block">the data buffer to be filled</param>
/// <param name="value">a value to be put in the p_Block buffer</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename T>
__global__ void kernel_fill_array(T *p_Block, T value, size_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		p_Block[i] = value;
	}
}

/// A class wrapping a custom memory allocator for fast memory operations on CUDA
/// This class provides means for allocation and deallocation of CUDA buffers
/// as well as data movement between host and device memory
/// Convinience methods are provided for filling the buffers with predefined data
class GPUMemory {
public:

    /// Struct for polygon column (with pointers to start of condensed buffers).
    /// A structure representing the data type of a complex polygon used
    /// for representing polygons and holes in them. This data structure is
    /// used during point in polygon and polygon intersect operations. It is the result
    /// of polygon intersect operations.
	struct GPUPolygon
	{
		/// Points of polygons
		NativeGeoPoint* polyPoints;
		/// Start indices of each polygon in point array
		int32_t* pointIdx;
		/// Number of points of each polygon
		int32_t* pointCount;
		/// Start indices of each complex polygon in polygon array
		int32_t* polyIdx;
		/// Number of polygons of each complex polygon
		int32_t* polyCount;
	};

	/// Struct for GPU representation of string column (with pointers to start of condensed buffers).
	struct GPUString
	{
		/// All chars from all strings condensed
		char * allChars;
		/// Start indices of each string in allChars array,
		/// shifted by 1 string to left (last one is total count of chars)
		int64_t * stringIndices;
	};

	static bool EvictWithLockList();

	/// Memory allocation of data blocks(buffers) on the GPU with the respective size of the input parameter type
	/// <param name="p_Block">pointer to pointer wich will points to allocated memory block on the GPU</param>
	/// <param name="dataElementCount">count of elements in the block with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void alloc(T **p_Block, size_t dataElementCount)
	{
		bool allocOK = false;
		while (!allocOK)
		{
			try
			{
				*p_Block = reinterpret_cast<T*>(Context::getInstance().GetAllocatorForCurrentDevice().allocate(dataElementCount * sizeof(T)));
				allocOK = true;
			}
			catch (const std::out_of_range& e)
			{
				if (!EvictWithLockList())
				{
					std::rethrow_exception(std::current_exception());
				}
			}
		}
		CheckCudaError(cudaGetLastError());
	}

	/// Synchronous memory allocation and setting of data blocks(buffers) on the GPU with the respective 
	/// size of the input parameter type
	/// <param name="p_Block">pointer to pointer wich will points to allocated memory block on the GPU</param>
	/// <param name="value">value to set the memory to
	/// (always has to be int, because of cudaMemset; and just lowest byte will be used
	/// and all bytes in the allocated buffer will be set to that byte value) 
	/// e.g. min: 0, max: 255, only these values are valid </param>
    /// <param name="dataElementCount">count of elements in the block with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void allocAndSet(T **p_Block, int value, size_t dataElementCount)
	{
		*p_Block = reinterpret_cast<T*>(Context::getInstance().GetAllocatorForCurrentDevice().allocate(dataElementCount * sizeof(T)));

		memset(*p_Block, value, dataElementCount);

		CheckCudaError(cudaGetLastError());
	}

	/// Asynchronous memory allocation and setting of data blocks(buffers) on the GPU with the
    /// respective size of the input parameter type 
	///<param name="p_Block">pointer to pointer wich will points to allocated memory block on the GPU</param> 
	/// <param name="value">value to set the memory to (always has to be int, because of cudaMemset; and 
	/// just lowest byte will be used and all bytes in the allocated buffer will be set to that byte value) 
	/// e.g. min: 0, max: 255, only these values are valid </param> 
	/// <param name="dataElementCount">count of elements in the block with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void memset(T *p_Block, int value, size_t dataElementCount)
	{
		cudaMemsetAsync(p_Block, value, dataElementCount * sizeof(T));
		CheckCudaError(cudaGetLastError());
	}
	#ifdef __CUDACC__
	template<typename T>
	static void fillArray(T *p_Block, T value, size_t dataElementCount)
	{
		kernel_fill_array << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(p_Block, value, dataElementCount);

		CheckCudaError(cudaGetLastError());
	}
	#endif

	/// Moving data from host to device
	/// Copy a memory block with dataType numbers from host (RAM, CPU's memory) to device (GPU's memory). 
	/// <param name="p_BlockDevice">pointer to destination device memory</param>
    /// <param name="p_BlockHost">pointer to source host memory</param>
    /// <param name="dataElementCount"> count of elements in the blocks with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void copyHostToDevice(T *p_BlockDevice, T *p_BlockHost, size_t dataElementCount)
	{
		cudaMemcpyAsync(p_BlockDevice, p_BlockHost, dataElementCount * sizeof(T), cudaMemcpyHostToDevice);
		CheckCudaError(cudaGetLastError());
	}

	/// Moving data from device to host
    /// Copy a memory block with dataType numbers from device (GPU's memory) to host (RAM, CPU's memory). 
	/// <param name="p_BlockHost">pointer to destination host memory</param>
    /// <param name="p_BlockDevice">pointer to source device memory</param>
    /// <param name="dataElementCount"> count of elements in the blocks with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void copyDeviceToHost(T *p_BlockHost, T *p_BlockDevice, size_t dataElementCount)
	{
		cudaMemcpy(p_BlockHost, p_BlockDevice, dataElementCount * sizeof(T), cudaMemcpyDeviceToHost);
		CheckCudaError(cudaGetLastError());
	}

	/// Moving data between buffers on the device
    /// Copy a memory block with dataType numbers from host (RAM, CPU's memory) to device (GPU's memory).
    /// <param name="p_BlockDestination">pointer to destination device memory</param>
    /// <param name="p_BlockSource">pointer to source device memory</param>
    /// <param name="dataElementCount"> count of elements in the blocks with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void copyDeviceToDevice(T *p_BlockDestination, T *p_BlockSource, size_t dataElementCount)
	{
		cudaMemcpy(p_BlockDestination, p_BlockSource, dataElementCount * sizeof(T), cudaMemcpyDeviceToDevice);
		CheckCudaError(cudaGetLastError());
	}

	/// Freeing data
	/// Free memory block from GPU's memory
	/// <param name="p_Block">pointer to a memory block in GPU memory</param>
	static void free(void *p_block)
	{
		Context::getInstance().GetAllocatorForCurrentDevice().deallocate(static_cast<int8_t*>(p_block), 0);
		CheckCudaError(cudaGetLastError());
	}

	/// Register a piece of unpaged host memory to be used for fast memory transfers between host and device
    /// < param name="devicePtr">pointer to device memory to be mapped</param> 
	/// < param name="hostPtr">pointer to host memory to be mapped into</param>
    /// <param name="dataElementCount"> count of elements to be mapped with size sizeof(T)*dataElementCount</param>
	template<typename T>
	static void hostRegister(T **devicePtr, T *hostPtr, size_t dataElementCount)
	{
		cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterMapped);
		cudaHostGetDevicePointer(devicePtr, hostPtr, 0);

		CheckCudaError(cudaGetLastError());
	}

	/// Un-Register a piece of unpaged host memory to be used for fast memory transfers between host and device
    /// < param name="hostPtr">pointer to host memory to be unmapped/freed into</param>
	template<typename T>
	static void hostUnregister(T *hostPtr)
	{
		cudaHostUnregister(hostPtr);
		CheckCudaError(cudaGetLastError());
	}

	// Pin host memory
	template<typename T>
	static void hostPin(T* hostPtr, size_t dataElementCount)
	{
		cudaHostRegister(hostPtr, dataElementCount * sizeof(T), cudaHostRegisterDefault);
		CheckCudaError(cudaGetLastError());
	}

	/// Clear all custom allocated memory with the memory allocator.
	/// This is a O(1) operation
	static void clear()
	{
		Context::getInstance().GetAllocatorForCurrentDevice().Clear();
		CheckCudaError(cudaGetLastError());
	}
};
