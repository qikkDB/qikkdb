#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "../NullConstants.cuh"

/// Namespace for unary string operation generic functors
namespace StringUnaryOperations
{
	struct ltrim
	{
		typedef GPUMemory::GPUString returnType;
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct rtrim
	{
		typedef GPUMemory::GPUString returnType;
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct lower
	{
		typedef GPUMemory::GPUString returnType;
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct upper
	{
		typedef GPUMemory::GPUString returnType;
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct reverse
	{
		typedef GPUMemory::GPUString returnType;
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct len
	{
		typedef int64_t returnType;
		__device__ int64_t operator()(GPUMemory::GPUString a) const
		{
			return 0;
		}
	};
}


template<typename OP>
__global__ void kernel_string_unary(GPUMemory::GPUString output, GPUMemory::GPUString ACol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		//TODO
	}
}

class GPUStringUnary
{
public:
	template<typename OP>
	static void col(GPUMemory::GPUString &output, GPUMemory::GPUString ACol, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_string_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, dataElementCount);
	}

	template<typename OP>
	static void cnst(GPUMemory::GPUString &output, GPUMemory::GPUString AConst, int32_t dataElementCount)
	{
		ErrorFlagSwapper errorFlagSwapper;
		kernel_string_unary <OP>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, dataElementCount);
	}
};