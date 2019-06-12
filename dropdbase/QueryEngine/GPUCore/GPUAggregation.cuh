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


/// Aggregation functions on the whole column
class GPUAggregation {
public:
	/// AGG=min: Find the smallest element in the collumn
	/// AGG=max: Find the largest element in the collumn
	/// AGG=sum: Return the sum of elements in the collumn
	/// AGG=avg: Return the average of elements in the collumn as a floating point number
	/// AGG=cnt: Return the number of elements in the collumn
	/// <param name="outValue">the smallest element- kept on the GPU</param>
	/// <param name="ACol">block of the the input collumn</param>
	/// <param name="dataType">input data type</param>
	/// <param name="dataElementCount">the size of the input blocks in bytes</param>
	/// <returns>GPU_EXTENSION_SUCCESS if operation was successful
	/// or GPU_EXTENSION_ERROR if some error occured</returns>
	template<typename AGG, typename OUT, typename IN>
	static void col(OUT *outValue, IN *ACol, int32_t dataElementCount)
	{
		// Functor call
		AGG::template agg<OUT, IN>(outValue, ACol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}
};

