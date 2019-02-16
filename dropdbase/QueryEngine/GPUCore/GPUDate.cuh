#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "MaybeDeref.cuh"

namespace DateOperations
{
	struct year
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
		}
	};

	struct month
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
		}
	};

	struct day
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
		}
	};

	struct hour
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
		}
	};

	struct minute
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
		}
	};

	struct second
	{
		template<typename RET, typename DT>
		__device__ RET operator()(DT dateTime) const
		{
			return 0;
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
template<typename OP, typename RET, typename DT>
__global__ void kernel_extract(RET * output, DT dateTimeCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		output[i] = OP{}.template operator()
			< RET, typename std::remove_pointer<DT>::type>
			(maybe_deref(dateTimeCol, i));
	}
}

/// <summary>
/// GPUDate class is for extracting (conversion) variables (e.g. days, hours)
/// from datetime column or constant
/// </summary>
class GPUDate
{
public:
	template<typename OP, typename T>
	static void extractCol(int32_t * output, T * dateTimeCol, int32_t dataElementCount)
	{
		kernel_extract <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, dateTimeCol, dataElementCount);
		cudaDeviceSynchronize();
		QueryEngineError::setCudaError(cudaGetLastError());
	}

	template<typename OP, typename T>
	static void extractConst(int32_t * output, T dateTimeConst, int32_t dataElementCount)
	{
		kernel_extract <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, dateTimeConst, dataElementCount);
		cudaDeviceSynchronize();
		QueryEngineError::setCudaError(cudaGetLastError());
	}


};
