#pragma once


#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaybeDeref.cuh"

namespace StringBinaryOperations
{
	struct left
	{
		typedef GPUMemory::GPUString returnType;
		template<typename T>
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a, T b) const
		{
			return GPUMemory::GPUString();
		}
	};

	struct right
	{
		typedef GPUMemory::GPUString returnType;
		template<typename T>
		__device__ GPUMemory::GPUString operator()(GPUMemory::GPUString a, T b) const
		{
			return GPUMemory::GPUString();
		}
	};
}


template<typename OP, typename T>
__global__ void kernel_string_binary(GPUMemory::GPUString output, GPUMemory::GPUString ACol, T BCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		//TODO
	}
}

class GPUStringBinary
{
public:
	template<typename OP, typename T>
	static void ColCol(GPUMemory::GPUString &output, GPUMemory::GPUString ACol, T BCol, int32_t dataElementCount)
	{
		kernel_string_binary <OP, T>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BCol, dataElementCount);
	}

	template<typename OP, typename T>
	static void ColConst(GPUMemory::GPUString &output, GPUMemory::GPUString ACol, T BConst, int32_t dataElementCount)
	{
		kernel_string_binary <OP, T>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, ACol, BConst, dataElementCount);
	}

	template<typename OP, typename T>
	static void ConstCol(GPUMemory::GPUString &output, GPUMemory::GPUString AConst, T BCol , int32_t dataElementCount)
	{
		kernel_string_binary <OP, T>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, BCol, dataElementCount);
	}

	template<typename OP, typename T>
	static void ConstConst(GPUMemory::GPUString &output, GPUMemory::GPUString AConst, T BConst, int32_t dataElementCount)
	{
		kernel_string_binary <OP, T>
			<< < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(output, AConst, BConst, dataElementCount);
	}
};