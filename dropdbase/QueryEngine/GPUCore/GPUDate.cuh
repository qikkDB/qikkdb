#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../QueryEngineError.h"
#include "MaybeDeref.cuh"

namespace DateOperations
{
	struct year
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return 0;
		}
	};

	struct month
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return 0;
		}
	};

	struct day
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return 0;
		}
	};

	struct hour
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return static_cast<int32_t>((dateTime/3600i64) % 24i64);
		}
	};

	struct minute
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return static_cast<int32_t>((dateTime / 60i64) % 60i64);
		}
	};

	struct second
	{
		__device__ int32_t operator()(int64_t dateTime) const
		{
			return static_cast<int32_t>(dateTime % 60i64);
		}
	};
}


/// <summary>
/// Kernel for extracting date or time variable (e.g. days, hours)
/// from datetime column or constant
/// </summary>
/// <param name="output">block of the result data</param>
/// <param name="dateTimeCol">input timestamp (column or constant)</param>
/// <param name="dataElementCount">the count of elements in the input block
/// (or of output block if input is constant)</param>
template<typename OP, typename T>
__global__ void kernel_extract(int32_t * output, T dateTimeCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = OP{}.operator()(maybe_deref(dateTimeCol, i));
	}
}

/// <summary>
/// GPUDate class is for extracting (conversion) variables (e.g. days, hours)
/// from datetime column or constant
/// </summary>
class GPUDate
{
public:
	template<typename OP>
	static void extractCol(int32_t * output, int64_t * dateTimeCol, int32_t dataElementCount)
	{
		kernel_extract <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, dateTimeCol, dataElementCount);
		cudaDeviceSynchronize();
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename OP>
	static void extractConst(int32_t * output, int64_t dateTimeConst, int32_t dataElementCount)
	{
		kernel_extract <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, dateTimeConst, dataElementCount);
		cudaDeviceSynchronize();
		QueryEngineError::setCudaError(cudaGetLastError());
	}

};
