#ifndef GPU_GROUP_BY_CUH
#define GPU_GROUP_BY_CUH

#include <cstdint>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"

#include "ErrorFlagSwapper.h"

constexpr int32_t EMPTY = -1;

// Generic agg function functors
namespace AggregationFunctions
{
	struct min
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			atomicMin(a, b);
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
		__device__ void operator()(T *a, T b) const
		{
			atomicMax(a, b);
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
		__device__ void operator()(T *a, T b) const
		{
			atomicAdd(a, b);
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
		__device__ void operator()(T *a, T b) const
		{
			atomicAdd(a, b);
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
		__device__ void operator()(T *a, T b) const
		{
			// empty
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
	int32_t maxHashCount,
	K *inKeys,
	V *inValues,
	int32_t dataElementCount,
	int32_t *errorFlag) {

	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		// Linear probing
		bool insertionSucceeded = false;
		for (int j = 0; j < maxHashCount; j++) {
			// Calculate hash - use type conversion because of float
			int32_t hashIndex = static_cast<int32_t>(abs((keys[i] + j))) % maxHashCount;

			// Check if a place is empty for the key to insert into the hash table
			if (indexTable[hashIndex] == EMPTY)
			{
				int32_t old = atomicCAS(&indexTable[hashIndex], EMPTY, *resultElementCount);
				if (old != EMPTY || old != *resultElementCount)
				{
					continue;
				}

				// Add a new key and increment the table size
				atomicExch(&keys[indexTable[hashIndex]], inKeys[i]);
				atomicAdd(resultElementCount, 1);
			}
			else if (keys[indexTable[hashIndex]] != inKeys[i])
			{
				continue;
			}

			// Insertion succeeded
			insertionSucceeded = true;

			// Atomic value modification based on agg function
			AGG{}(&values[indexTable[hashIndex]], inValues[i]);
			atomicAdd(&keyOccurenceCount[indexTable[hashIndex]], 1);
		}

		// Set error flag if linear probing failed - hash table full
		if (insertionSucceeded == false) {
			atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_HASH_TABLE_FULL));
			break;
		}
	}
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

	int32_t maxHashCount_;			// Maximum size of the result hash table

	ErrorFlagSwapper errorFlagSwapper_;
public:


	// Constructor
	// Allocates hash table of element count: hashHashCount
	GPUGroupBy(int32_t maxHashCount) :
		maxHashCount_(maxHashCount)
	{
		GPUMemory::alloc(&keys_, hashElementCount);
		GPUMemory::allocAndSet(&values_, AGG::getInitValue<V>(), hashElementCount);
		GPUMemory::allocAndSet(&keyOccurenceCount_, 0, hashElementCount);
		GPUMemory::allocAndSet(&indexTable_, EMPTY, hashElementCount);
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

	// Ge the count of accumulated hash entries so far for result buffer allocation
	int32_t getResultElementCount() {
		int32_t temp;
		GPUMemory::copyDeviceToHost(&temp, resultElementCount_, 1);
		return temp;
	}

	// Group By - callable on the blocks of the input dataset
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		group_by_kernel <AGG> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(keys_, values_, keyOccurenceCount_, indexTable_, resultElementCount_, maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.getFlagPointer());
	}

	// Get the final hash table results - buffers need to be pre allocated
	void getResults(K *outKeys, V *outValues)
	{
		int32_t resultElementCount = getResultElementCount();

		// Copy back the keys
		GPUMemory::copyDeviceToHost(outKeys, keys_, resultElementCount);

		// Copy back the results based on the operation
		if (std::is_same < AGG, min>::value || std::is_same < AGG, max>::value || std::is_same < AGG, sum>::value)
		{
			GPUMemory::copyDeviceToHost(outValues, values_, resultElementCount);
		}
		else if (std::is_same < AGG, avg>::value)
		{
			GPUArithmetic::division(values_, values_, keyOccurenceCount_, resultElementCount);
			GPUMemory::copyDeviceToHost(outValues, values_, resultElementCount);
		}
		else if (std::is_same < AGG, cnt>::value)
		{
			GPUMemory::copyDeviceToHost(outValues, keyOccurenceCount_, resultElementCount);
		}
		else
		{
			int32_t temp = QueryEngineError::GPU_UNKNOWN_AGG_FUN;
			GPUMemory::copyHostToDevice(errorFlagSwapper_.getFlagPointer(), &temp, 1);
			return;
		}
	}
};

#endif