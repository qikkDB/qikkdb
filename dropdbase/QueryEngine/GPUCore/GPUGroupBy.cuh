#ifndef GPU_GROUP_BY_CUH
#define GPU_GROUP_BY_CUH

#include <cstdint>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "GPUMemory.cuh"

#include "ErrorFlagSwapper.h"

// Generic agg function functors
namespace AggregationFunctions
{
	struct min
	{
		template<typename T>
		__device__ T operator()(T a, T b) const
		{
			return a < b ? a : b;
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return std::numeric_limits<T>::max();
		}
	};

	struct max
	{
		template<typename T>
		__device__ T operator()(T a, T b) const
		{
			return a > b ? a : b;
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return std::numeric_limits<T>::lowest();
		}
	};

	struct sum
	{
		template<typename T>
		__device__ T operator()(T a, T b) const
		{
			return a + b;
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};

	struct avg
	{
		template<typename T>
		__device__ T  operator()(T a, T b) const
		{
			return a + b;
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};
	struct cnt
	{
		template<typename T>
		__device__ T  operator()(T a, T b) const
		{

		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};
}

template<typename AGG, typename K, typename V>
__global__ void group_by_kernel(
	K *keys,
	V *values,
	int32_t *keyOccurenceCount,
	int32_t *indexTable,
	int32_t *resultElementCount,
	K *inKeys,
	V *inValues,
	int32_t dataElementCount,
	int32_t *errorFlag)
{

}

template<typename AGG, typename K, typename V>
class GPUGroupBy
{
private:
	K *keys_;						// Keys
	V *values_;						// Values
	int32_t *keyOccurenceCount_;	// Count of occurrances of keys
	int32_t *indexTable_;			// Table for indexing keys to avoid reconstruct operation
	int32_t *resultElementCount_;	// Counter that counts the size of the result hash table - incremented atomically

	int32_t maxElementCount_;		// Maximum size of the result hash table

	ErrorFlagSwapper errorFlagSwapper_;
public:


	// Constructor
	// Allocates hash table of element count: hashElementCount
	GPUGroupBy(int32_t hashElementCount) :
		maxElementCount_(hashElementCount)
	{
		GPUMemory::alloc(&keys_, hashElementCount);
		GPUMemory::allocAndSet(&values_, AGG::getInitValue<V>(), hashElementCount);
		GPUMemory::allocAndSet(&keyOccurenceCount_, 0, hashElementCount);
		GPUMemory::allocAndSet(&indexTable_, -1, hashElementCount);
		GPUMemory::allocAndSet(&resultElementCount_, 0, 1);
	}

	// Destructor
	~GPUGroupBy()
	{
		GPUMemory::free(keys_);
		GPUMemory::free(values_);
		GPUMemory::free(keyOccurenceCount_);
		GPUMemory::free(indexTable_);
		GPUMemory::free(resultElementCount_);
	}

	GPUGroupBy(const GPUGroupBy &) = delete;
	GPUGroupBy& operator=(const GPUGroupBy &) = delete;

	// Group By - callable on the blocks of the input dataset
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		group_by_kernel <AGG> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(keys_, values_, keyOccurenceCount_, indexTable_, resultElementCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.getFlagPointer());
	}

	// Get the final hash table results
	void getResults()
	{
		// TODO Return results
		// TODO calculate avg or return count
	}
};

#endif