#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"


template <typename OP>
__global__ void kernel_per_char_unary(char* outChars, char* inChars, int64_t charCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < charCount; i += stride)
	{
		outChars[i] = OP{}(inChars[i]);
	}
}


/// Namespace for implementation of unary string operations on GPU
/// Hierarchy: Length Variabilities -> Operations
namespace StringUnaryOpHierarchy
{
	struct variable
	{
		typedef GPUMemory::GPUString returnType;
		template <typename OP>
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			return GPUMemory::GPUString(); // TODO
		}
	};

	struct fixed
	{
		typedef GPUMemory::GPUString returnType;
		template <typename OP>
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			GPUMemory::GPUString outCol;
			if (inputIsCol)	// Col
			{
				GPUMemory::alloc(&(outCol.stringIndices), outStringCount);
				GPUMemory::copyDeviceToDevice(outCol.stringIndices, input.stringIndices, outStringCount);
				int64_t totalCharCount;
				GPUMemory::copyDeviceToHost(&totalCharCount, input.stringIndices + outStringCount - 1, 1);
				GPUMemory::alloc(&(outCol.allChars), totalCharCount);
				kernel_per_char_unary<OP> << <Context::getInstance().calcGridDim(totalCharCount),
					Context::getInstance().getBlockDim() >> >
					(outCol.allChars, input.allChars, totalCharCount);
				CheckCudaError(cudaGetLastError());
			}
			else	// Const (expand 1 const result to col)
			{
				// TODO
			}
			return outCol;
		}
	};

	/// Namespace for variable length unary operations
	namespace VariableLength
	{
		struct ltrim
		{

		};

		struct rtrim
		{

		};
	} // namespace VariableLength

	/// Namespace for fixed length unary operations (per-char unary operations)
	namespace FixedLength
	{
		struct lower
		{
			__device__ char operator()(char c) const
			{
				return (c >= 'a' && c <= 'z')? (c & 0xDF) : c;
			}
		};

		struct upper
		{
			__device__ char operator()(char c) const
			{
				return (c >= 'A' && c <= 'Z') ? (c | 0x20) : c;
			}
		};
	} // namespace FixedLength

} // namespace StringUnaryOperations


/// Namespace for unary string operation generic functors
namespace StringUnaryOperations
{
	struct ltrim
	{
		typedef GPUMemory::GPUString returnType;
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			return StringUnaryOpHierarchy::variable{}.template operator()
				< StringUnaryOpHierarchy::VariableLength::ltrim >
				(outStringCount, input, inputIsCol);
		}
	};

	struct rtrim
	{
		typedef GPUMemory::GPUString returnType;
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			return StringUnaryOpHierarchy::variable{}.template operator()
				< StringUnaryOpHierarchy::VariableLength::rtrim >
				(outStringCount, input, inputIsCol);
		}
	};

	struct lower
	{
		typedef GPUMemory::GPUString returnType;
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			return StringUnaryOpHierarchy::fixed{}.template operator()
				< StringUnaryOpHierarchy::FixedLength::lower >
				(outStringCount, input, inputIsCol);
		}
	};

	struct upper
	{
		typedef GPUMemory::GPUString returnType;
		GPUMemory::GPUString operator()(int32_t outStringCount,
			GPUMemory::GPUString input, bool inputIsCol) const
		{
			return StringUnaryOpHierarchy::fixed{}.template operator()
				< StringUnaryOpHierarchy::FixedLength::upper >
				(outStringCount, input, inputIsCol);
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


class GPUStringUnary
{
public:
    template <typename OP>
    static void Col(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, int32_t dataElementCount)
    {
        output = OP{}(dataElementCount, ACol, true);
    }

    template <typename OP>
    static void Const(GPUMemory::GPUString& output, GPUMemory::GPUString AConst, int32_t dataElementCount)
    {
		output = OP{}(dataElementCount, AConst, false);
    }
};
