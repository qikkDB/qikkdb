#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"
#include "../../GpuSqlParser/DispatcherCpu/CpuFilterInterval.h"

/// Functors for parallel binary filtration operations
namespace FilterConditions
{
	/// A greater than operator > functor
	struct greater
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a > b;
		}
	};

	/// A greater than or equal operator >= functor
	struct greaterEqual
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a >= b;
		}
	};

	/// A less than operator < functor
	struct less
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a < b;
		}
	};

	/// A less than or equal operator <= functor
	struct lessEqual
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::NONE;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a <= b;
		}
	};

	/// An equality operator == functor
	struct equal
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::INNER;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a == b;
		}
	};

	/// An unequality operator != functor
	struct notEqual
	{
		static constexpr CpuFilterInterval interval = CpuFilterInterval::OUTER;
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a != b;
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for comparing values
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataElementCount">the count of elements in the input block</param>
template<typename OP, typename T, typename U>
__global__ void kernel_filter(int8_t *outMask, T ACol, U BCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = OP{}.template operator()
			< typename std::remove_pointer<T>::type, typename std::remove_pointer<U>::type >
			(maybe_deref(ACol, i), maybe_deref(BCol, i));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// A class for column-wise logic operation for data filtering based on logic conditions
class GPUFilter
{
public:
    /// Filtration operation with two columns of binary masks
    /// <param name="OP">Template parameter for the choice of the filtration operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void colCol(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
	{
		kernel_filter <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

    /// Filtration operation between a column of binary values and a binary value constant
    /// <param name="OP">Template parameter for the choice of the filtration operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="ACol">buffer with left side operands</param>
    /// <param name="BConst">right side constant operand</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void colConst(int8_t *outMask, T *ACol, U BConst, int32_t dataElementCount)
	{
		kernel_filter <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BConst, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

    /// Filtration operation between a column of binary values and a binary value constant
    /// <param name="OP">Template parameter for the choice of the filtration operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="AConst">left side constant operand</param>
    /// <param name="BCol">buffer with right side operands</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void constCol(int8_t *outMask, T AConst, U *BCol, int32_t dataElementCount)
	{
		kernel_filter <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, BCol, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Filtration operation betweentwo binary constants
    /// <param name="OP">Template parameter for the choice of the filtration operation</param>
    /// <param name="outMask">output GPU buffer mask</param>
    /// <param name="AConst">left side constant operand</param>
    /// <param name="BConst">right side constant operand</param>
    /// <param name="dataElementCount">data element count of the input block</param>
	template<typename OP, typename T, typename U>
	static void constConst(int8_t *outMask, T AConst, U BConst, int32_t dataElementCount)
	{
		GPUMemory::memset(outMask, OP{}(AConst, BConst), dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

};

