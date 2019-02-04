#ifndef GPU_AGGREGATION_CUH
#define GPU_AGGREGATION_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "../Context.h"
#include "../CudaMemAllocator.h"
#include "GPUMemory.cuh"


class GPUAggregation {
public:
	// Aggregation functions on the whole collumn
	/// <summary>
	/// AGG=min: Find the smallest element in the collumn
	/// AGG=max: Find the largest element in the collumn
	/// AGG=sum: Return the sum of elements in the collumn
	/// AGG=avg: Return the average of elements in the collumn as a floating point number
	/// AGG=cnt: Return the number of elements in the collumn
	/// </summary>
	/// <param name="outValue">the smallest element- kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename AGG, typename T>
	static void col(T *outValue, T *ACol, int32_t dataElementCount)
	{
		// Kernel call
		AGG::template agg<T>(outValue, ACol, dataElementCount);

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif 

