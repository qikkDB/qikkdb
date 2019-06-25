#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "MaybeDeref.cuh"
#include "../../GpuSqlParser/DispatcherCpu/CpuFilterInterval.h"
#include "GPUStringUnary.cuh"

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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			if (aLength != bLength)
			{
				return false;
			}
			else
			{
				for (int32_t j = 0; j < aLength; j++)
				{
					if (a[j] != b[j])
					{
						return false;
					}
				}
				return true;
			}
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

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			if (aLength != bLength)
			{
				return true;
			}
			else
			{
				for (int32_t j = 0; j < aLength; j++)
				{
					if (a[j] != b[j])
					{
						return true;
					}
				}
				return false;
			}
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

/// Kernel for string comparison (equality, ...)
template<typename OP>
__global__ void kernel_filter_string(int8_t *outMask, GPUMemory::GPUString inputA, bool isACol,
	GPUMemory::GPUString inputB, bool isBCol, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		const int32_t aI = isACol ? i : 0;
		const int64_t aIndex = GetStringIndex(inputA.stringIndices, aI);
		const int32_t aLength = static_cast<int32_t>(inputA.stringIndices[aI] - aIndex);
		const int32_t bI = isBCol ? i : 0;
		const int64_t bIndex = GetStringIndex(inputB.stringIndices, bI);
		const int32_t bLength = static_cast<int32_t>(inputB.stringIndices[bI] - bIndex);
		outMask[i] = OP{}.compareStrings(inputA.allChars + aIndex, aLength, inputB.allChars + bIndex, bLength);
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


	/// Filtration operation between two strings (column-column)
	template<typename OP>
	static void colCol(int8_t *outMask, GPUMemory::GPUString ACol, GPUMemory::GPUString BCol, int32_t dataElementCount)
	{
		kernel_filter_string <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, true, BCol, true, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Filtration operation between two strings (column-constant)
	template<typename OP>
	static void colConst(int8_t *outMask, GPUMemory::GPUString ACol, GPUMemory::GPUString BConst, int32_t dataElementCount)
	{
		kernel_filter_string <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, true, BConst, false, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Filtration operation between two strings (constant-column)
	template<typename OP>
	static void constCol(int8_t *outMask, GPUMemory::GPUString AConst, GPUMemory::GPUString BCol, int32_t dataElementCount)
	{
		kernel_filter_string <OP> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, false, BCol, true, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

	/// Filtration operation between two strings (constant-constant)
	template<typename OP>
	static void constConst(int8_t *outMask, GPUMemory::GPUString AConst, GPUMemory::GPUString BConst, int32_t dataElementCount)
	{
		// Compare constants
		kernel_filter_string <OP> << < Context::getInstance().calcGridDim(1), Context::getInstance().getBlockDim() >> >
			(outMask, AConst, false, BConst, false, 1);
		CheckCudaError(cudaGetLastError());

		// Expand mask - copy the one result to whole mask
		int8_t numberFromMask;
		GPUMemory::copyDeviceToHost(&numberFromMask, outMask, 1);
		GPUMemory::memset(outMask, numberFromMask, dataElementCount);
	}
};
