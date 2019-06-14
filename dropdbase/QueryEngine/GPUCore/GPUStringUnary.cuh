#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "GPUMemory.cuh"
#include "GPUReconstruct.cuh"
#include "cuda_ptr.h"


template <typename OP>
__global__ void kernel_predict_length_xtrim(int32_t * newLengths, GPUMemory::GPUString inCol, int32_t stringCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < stringCount; i += stride)
	{
		const int64_t index = (i == 0) ? 0 : inCol.stringIndices[i - 1];
		const int32_t length = static_cast<int32_t>(inCol.stringIndices[i] - index);
		int32_t j = 0;
		while (inCol.allChars[index + OP::GetIndex(j, length)] == ' ' && j < length)
		{
			j++;
		}
		newLengths[i] = length - j;
	}
}

template <typename OP>
__global__ void kernel_string_xtrim(GPUMemory::GPUString outCol, GPUMemory::GPUString inCol, int32_t stringCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < stringCount; i += stride)
	{
		const int64_t inIndex = (i == 0) ? 0 : inCol.stringIndices[i - 1];
		const int32_t inLength = static_cast<int32_t>(inCol.stringIndices[i] - inIndex);
		const int64_t outIndex = (i == 0) ? 0 : outCol.stringIndices[i - 1];
		const int32_t outLength = static_cast<int32_t>(outCol.stringIndices[i] - outIndex);
		const int64_t inStart = inIndex + OP::GetOffset(inLength, outLength);
		for (int32_t j = 0; j < outLength; j++)
		{
			outCol.allChars[outIndex + j] = inCol.allChars[inStart + j];
		}
	}
}

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

__global__ void kernel_reverse_string(GPUMemory::GPUString outCol, GPUMemory::GPUString inCol, int32_t stringCount);

/// Namespace for implementation of unary string operations on GPU
/// Hierarchy: Length Variabilities -> Operations
namespace StringUnaryOpHierarchy
{
	/// Namespace for variable length unary operations
	namespace VariableLength
	{
		struct ltrim
		{
			__device__ static const int32_t GetIndex(int32_t j, int32_t length)
			{
				return j;	// normal order of finding spaces
			}

			__device__ static const int32_t GetOffset(int32_t inLength, int32_t outLength)
			{
				return inLength - outLength;	// offset on string start
			}
		};

		struct rtrim
		{
			__device__ static const int32_t GetIndex(int32_t j, int32_t length)
			{
				return length - 1 - j;	// reverse order of finding spaces
			}

			__device__ static const int32_t GetOffset(int32_t inLength, int32_t outLength)
			{
				return 0;	// no offset on string start
			}
		};
	} // namespace VariableLength

	/// Namespace for fixed length unary operations (per-char unary operations)
	namespace FixedLength
	{
		struct lower
		{
			__device__ char operator()(char c) const
			{
				return (c >= 'A' && c <= 'Z') ? (c | 0x20) : c;
			}
		};

		struct upper
		{
			__device__ char operator()(char c) const
			{
				return (c >= 'a' && c <= 'z')? (c & 0xDF) : c;
			}
		};

		struct reverse
		{
			// no function needed
		};

	} // namespace FixedLength

	struct variable
	{
		template <typename OP>
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			GPUMemory::GPUString outCol;
			if (stringCount > 0)
			{
				Context& context = Context::getInstance();
				// Predict new lengths
				cuda_ptr<int32_t> newLengths(stringCount);
				kernel_predict_length_xtrim <OP> << <context.calcGridDim(stringCount),
					context.getBlockDim() >> >
					(newLengths.get(), input, stringCount);

				// Calculate new indices
				GPUMemory::alloc(&(outCol.stringIndices), stringCount);
				GPUReconstruct::PrefixSum(outCol.stringIndices, newLengths.get(), stringCount);

				// Do the xtrim ('x' will be l or r) by copying chars
				int64_t newTotalCharCount;
				GPUMemory::copyDeviceToHost(&newTotalCharCount, outCol.stringIndices + stringCount - 1, 1);
				GPUMemory::alloc(&(outCol.allChars), newTotalCharCount);
				kernel_string_xtrim <OP> << <context.calcGridDim(stringCount),
					context.getBlockDim() >> >
					(outCol, input, stringCount);
				CheckCudaError(cudaGetLastError());
			}
			else
			{
				outCol.stringIndices = nullptr;
				outCol.allChars = nullptr;
			}
			return outCol;
		}
	};

	struct fixed
	{
		template <typename OP>
		static void CallKernel(GPUMemory::GPUString outCol, GPUMemory::GPUString input,
			int32_t stringCount, int64_t totalCharCount)
		{
			Context& context = Context::getInstance();
			kernel_per_char_unary<OP> << <context.calcGridDim(totalCharCount),
					context.getBlockDim() >> >
					(outCol.allChars, input.allChars, totalCharCount);
		}

		template <typename OP>
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			GPUMemory::GPUString outCol;
			if(stringCount > 0)
			{
				GPUMemory::alloc(&(outCol.stringIndices), stringCount);
				GPUMemory::copyDeviceToDevice(outCol.stringIndices, input.stringIndices, stringCount);
				int64_t totalCharCount;
				GPUMemory::copyDeviceToHost(&totalCharCount, input.stringIndices + stringCount - 1, 1);
				GPUMemory::alloc(&(outCol.allChars), totalCharCount);
				CallKernel<OP>(outCol, input, stringCount, totalCharCount);
				CheckCudaError(cudaGetLastError());
			}
			else
			{
				outCol.stringIndices = nullptr;
				outCol.allChars = nullptr;
			}
			return outCol;
		}
	};

	template <>
	void fixed::CallKernel<StringUnaryOpHierarchy::FixedLength::reverse>
				(GPUMemory::GPUString outCol, GPUMemory::GPUString input,
				int32_t stringCount, int64_t totalCharCount);

} // namespace StringUnaryOperations


/// Namespace for unary string to string operation generic functors
namespace StringUnaryOperations
{
	struct ltrim
	{
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			return StringUnaryOpHierarchy::variable{}.template operator()
				< StringUnaryOpHierarchy::VariableLength::ltrim >
				(input, stringCount);
		}
	};

	struct rtrim
	{
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			return StringUnaryOpHierarchy::variable{}.template operator()
				< StringUnaryOpHierarchy::VariableLength::rtrim >
				(input, stringCount);
		}
	};

	struct lower
	{
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			return StringUnaryOpHierarchy::fixed{}.template operator()
				< StringUnaryOpHierarchy::FixedLength::lower >
				(input, stringCount);
		}
	};

	struct upper
	{
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			return StringUnaryOpHierarchy::fixed{}.template operator()
				< StringUnaryOpHierarchy::FixedLength::upper >
				(input, stringCount);
		}
	};

	struct reverse
	{
		GPUMemory::GPUString operator()(GPUMemory::GPUString input, int32_t stringCount) const
		{
			return StringUnaryOpHierarchy::fixed{}.template operator()
				< StringUnaryOpHierarchy::FixedLength::reverse >
				(input, stringCount);
		}
	};
}

/// Namespace for unary string to int operation generic functors
namespace StringIntUnaryOperations
{
	/// Length of string
	struct len
	{
		// no function needed
	};
}


class GPUStringUnary
{
public:
    template <typename OP>
    static void Col(GPUMemory::GPUString& output, GPUMemory::GPUString ACol, int32_t dataElementCount)
    {
        output = OP{}(ACol, dataElementCount);
    }

    template <typename OP>
    static void Const(GPUMemory::GPUString& output, GPUMemory::GPUString AConst)
    {
		output = OP{}(AConst, 1);
    }
	
    template <typename OP>
	static void Col(int32_t * outCol, GPUMemory::GPUString inCol, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();
		kernel_lengths_from_indices << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outCol, inCol.stringIndices, dataElementCount);
		CheckCudaError(cudaGetLastError());
	}

    template <typename OP>
	static void Const(int32_t * outCol, GPUMemory::GPUString inCol)
	{
		// Copy single index to host
		int64_t hostIndex;
		GPUMemory::copyDeviceToHost(&hostIndex, inCol.stringIndices, 1);
		// Cast to int32 and copy to return result
		int32_t length = static_cast<int32_t>(hostIndex);
		GPUMemory::copyHostToDevice(outCol, &length, 1);
	}
};
