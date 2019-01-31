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
#include "GPUReconstruct.cuh"

#include "ErrorFlagSwapper.h"
#include "cuda_ptr.h"

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

		// Specialized atomicMin for floats
		__device__ void operator()(float *a, float b) const
		{
			float old = *a;
			float expected;
			if (old <= b)
			{
				return;
			}

			do
			{
				expected = old;
				int32_t ret = atomicCAS((int32_t*)a, *(int32_t*)(&expected), *(int32_t*)(&b));
				old = *(float*)&ret;
			} while (old != expected && old > b);
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

		// Specialized atomicMax for floats
		__device__ void operator()(float *a, float b) const
		{
			float old = *a;
			float expected;
			if (old >= b)
			{
				return;
			}

			do
			{
				expected = old;
				int32_t ret = atomicCAS((int32_t*)a, *(int32_t*)(&expected), *(int32_t*)(&b));
				old = *(float*)&ret;
			} while (old != expected && old < b);
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

// Universal null key calculator
template<typename K>
__device__ __host__ constexpr K getEmptyValue()
{
	static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
		"Unsupported data type of key (in function getEmptyValue)");

	if (std::is_integral<K>::value)
	{
		return std::numeric_limits<K>::min();
	}
	else if (std::is_floating_point<K>::value)
	{
		return std::numeric_limits<K>::quiet_NaN();
	}
}

// Kernel
template<typename AGG, typename K, typename V>
__global__ void group_by_kernel(
	K *keys,
	V *values,
	int64_t *keyOccurenceCount,
	int32_t maxHashCount,
	K *inKeys,
	V *inValues,
	int32_t dataElementCount,
	int32_t *errorFlag) {

	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		int32_t hash = static_cast<int32_t>(inKeys[i]); // TODO maybe improve hashing for float
		int32_t foundIndex = -1;
		for (int j = 0; j < maxHashCount; j++) {
			// Calculate hash - use type conversion because of float
			int32_t index = abs((hash + j) % maxHashCount);

			//Check if key is not empty and key is not equal to the currently inserted key
			if (keys[index] != getEmptyValue<K>() && keys[index] != inKeys[i])
			{
				continue;
			}

			// If key is empty
			if (keys[index] == getEmptyValue<K>())
			{
				// Compare key at index with Empty and if equals, store there inKey
				K old = atomicCAS(&keys[index], getEmptyValue<K>(), inKeys[i]);

				// Check if some other thread stored different key to this index
				if (old != getEmptyValue<K>() && old != inKeys[i])
				{
					continue;  // Try to find another index
				}

				/* // Explanation - all conditions explicitly defined
				if (old != getEmptyValue<K>())
				{
					if (old == inKeys[i])
					{
						foundIndex = index;
						printf("Existing key: %d\n", inKeys[i]);
						break;
					}
					else // old != inKeys[i]
					{
						printf("Lost race: %d\n", inKeys[i]);
						continue; // try to find another index
					}
				}
				else // old == getEmptyValue<K>()
				{
					if (keys[index] == inKeys[i])
					{
						foundIndex = index;
						printf("Added key: %d\n", inKeys[i]);
						break;
					}
					else // keys[index] != inKeys[i]
					{
						printf("this will never happen: %d\n", inKeys[i]);
						continue; // try to find another index
					}
				}*/
			}
			else if (keys[index] != inKeys[i])
			{
				continue;  // try to find another index
			}

			//The key was added or found as already existing
			foundIndex = index;
			break;
		}

		// If no index was found - the hash table is full 
		// else if we found a valid index
		if (foundIndex == -1)
		{
			atomicExch(errorFlag, static_cast<int32_t>(QueryEngineError::GPU_HASH_TABLE_FULL));
		}
		else
		{
			// Use aggregation of values on the bucket and the corresponding counter
			AGG{}(&values[foundIndex], inValues[i]);
			atomicAdd((uint64_t*)&keyOccurenceCount[foundIndex], 1);
		}
	}
}

template<typename K>
__global__ void is_bucket_occupied_kernel(int32_t *occupancyMask, K *keys, int32_t maxHashCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < maxHashCount; i += stride)
	{
		if (keys[i] == getEmptyValue<K>())
		{
			occupancyMask[i] = 0;
		}
		else
		{
			occupancyMask[i] = 1;
		}
	}
}

template<typename AGG, typename O, typename K, typename V>
class GPUGroupBy
{
private:
	K *keys_;							// Keys
	V *values_;							// Values
	int64_t *keyOccurenceCount_;		// Count of occurrances of keys		

	int32_t maxHashCount_;				// Maximum size of the result hash table

	ErrorFlagSwapper errorFlagSwapper_;

public:
	// Constructor
	// Allocates hash table of element count: hashHashCount
	GPUGroupBy(int32_t maxHashCount) :
		maxHashCount_(maxHashCount)
	{
		GPUMemory::alloc(&keys_, maxHashCount_);
		GPUMemory::alloc(&values_, maxHashCount_);
		GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);

		GPUMemory::fillArray(keys_, getEmptyValue<K>(), maxHashCount_);
		GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
	}

	// Destructor
	~GPUGroupBy()
	{
		GPUMemory::free(keys_);
		GPUMemory::free(values_);
		GPUMemory::free(keyOccurenceCount_);
	}

	GPUGroupBy(const GPUGroupBy &) = delete;
	GPUGroupBy& operator=(const GPUGroupBy &) = delete;

	// Group By - callable on the blocks of the input dataset
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		group_by_kernel <AGG> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(keys_, values_, keyOccurenceCount_, maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.getFlagPointer());
	}

	// Get the final hash table results - for operations Min, Max and Sum
	void getResults(K *outKeys, O *outValues, int32_t *outDataElementCount)
	{
		static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
			"GPUGroupBy<min/max/sum>.getResults K (keys) must be integral or floating point");
		static_assert(std::is_integral<V>::value || std::is_floating_point<V>::value,
			"GPUGroupBy<min/max/sum>.getResults V (values) must be integral or floating point");
		static_assert(std::is_same<O, V>::value,
			"GPUGroupBy<min/max/sum>.getResults O (outValue) and V (value) must be of the same type (for Min/Max/Sum)");

		// Create buffer for bucket compression - reconstruct
		cuda_ptr<int32_t> occupancyMask(maxHashCount_, 0);

		// Calculate occupancy mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask.get(), keys_, maxHashCount_);

		// Reconstruct the output
		// Copy back the results based on the operation
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask.get(), maxHashCount_);
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_, occupancyMask.get(), maxHashCount_);
	}
};

template<typename O, typename K, typename V>
class GPUGroupBy<AggregationFunctions::avg, O, K, V>
{
private:
	K *keys_;							// Keys
	V *values_;							// Values
	int64_t *keyOccurenceCount_;		// Count of occurrances of keys		

	int32_t maxHashCount_;				// Maximum size of the result hash table

	ErrorFlagSwapper errorFlagSwapper_;

public:
	// Constructor
	// Allocates hash table of element count: hashHashCount
	GPUGroupBy(int32_t maxHashCount) :
		maxHashCount_(maxHashCount)
	{
		GPUMemory::alloc(&keys_, maxHashCount_);
		GPUMemory::alloc(&values_, maxHashCount_);
		GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);

		GPUMemory::fillArray(keys_, getEmptyValue<K>(), maxHashCount_);
		GPUMemory::fillArray(values_, AggregationFunctions::avg::template getInitValue<V>(), maxHashCount_);
	}

	// Destructor
	~GPUGroupBy()
	{
		GPUMemory::free(keys_);
		GPUMemory::free(values_);
		GPUMemory::free(keyOccurenceCount_);
	}

	GPUGroupBy(const GPUGroupBy &) = delete;
	GPUGroupBy& operator=(const GPUGroupBy &) = delete;

	// Group By - callable on the blocks of the input dataset
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		group_by_kernel <AggregationFunctions::avg> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(keys_, values_, keyOccurenceCount_, maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.getFlagPointer());
	}

	// Get the final hash table results - for operation Average
	void getResults(K *outKeys, O *outValues, int32_t *outDataElementCount)
	{
		static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
			"GPUGroupBy<avg>.getResults K (keys) must be integral or floating point");
		static_assert(std::is_integral<V>::value || std::is_floating_point<V>::value,
			"GPUGroupBy<avg>.getResults V (values) must be integral or floating point");
		static_assert(std::is_floating_point<O>::value,
			"GPUGroupBy<avg>.getResults O (outValue) must be floating point for Average operation");

		// Create buffer for bucket compression - reconstruct
		cuda_ptr<int32_t> occupancyMask(maxHashCount_, 0);

		// Calculate occupancy mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask.get(), keys_, maxHashCount_);

		// TODO maybe somewhen optimize if O and V is the same data type - dont copy values
		//      but it requires one more GPUGroupBy specialization
		/*
		if (std::is_same<O, V>::value) {
			GPUArithmetic::division(values_, values_, keyOccurenceCount_, maxHashCount_);
			GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_, occupancyMask.get(), maxHashCount_);
		}
		else
		{
		*/
		cuda_ptr<O> outValuesGPU(maxHashCount_);
		// Divide by counts to get averages for buckets
		GPUArithmetic::division(outValuesGPU.get(), values_, keyOccurenceCount_, maxHashCount_);
		// Reonstruct result with original occupancyMask
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, outValuesGPU.get(), occupancyMask.get(), maxHashCount_);
		/*
		}
		*/
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask.get(), maxHashCount_);
	}
};

template<typename K, typename V>
class GPUGroupBy<AggregationFunctions::cnt, int64_t, K, V>
{
private:
	K *keys_;							// Keys
	V *values_;							// Values
	int64_t *keyOccurenceCount_;		// Count of occurrances of keys		

	int32_t maxHashCount_;				// Maximum size of the result hash table

	ErrorFlagSwapper errorFlagSwapper_;

public:
	// Constructor
	// Allocates hash table of element count: hashHashCount
	GPUGroupBy(int32_t maxHashCount) :
		maxHashCount_(maxHashCount)
	{
		GPUMemory::alloc(&keys_, maxHashCount_);
		GPUMemory::alloc(&values_, maxHashCount_);
		GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);

		GPUMemory::fillArray(keys_, getEmptyValue<K>(), maxHashCount_);
		GPUMemory::fillArray(values_, AggregationFunctions::cnt::template getInitValue<V>(), maxHashCount_);
	}

	// Destructor
	~GPUGroupBy()
	{
		GPUMemory::free(keys_);
		GPUMemory::free(values_);
		GPUMemory::free(keyOccurenceCount_);
	}

	GPUGroupBy(const GPUGroupBy &) = delete;
	GPUGroupBy& operator=(const GPUGroupBy &) = delete;

	// Group By - callable on the blocks of the input dataset
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		group_by_kernel <AggregationFunctions::cnt> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(keys_, values_, keyOccurenceCount_, maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.getFlagPointer());
	}

	// Get the final hash table results - for operation Count
	void getResults(K *outKeys, int64_t *outValues, int32_t *outDataElementCount)
	{
		static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
			"GPUGroupBy<cnt>.getResults K (keys) must be integral or floating point");
		static_assert(std::is_integral<V>::value || std::is_floating_point<V>::value,
			"GPUGroupBy<cnt>.getResults V (values) must be integral or floating point");

		// Create buffer for bucket compression - reconstruct
		cuda_ptr<int32_t> occupancyMask(maxHashCount_, 0);

		// Calculate occupancy mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask.get(), keys_, maxHashCount_);

		// Reconstruct the output
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask.get(), maxHashCount_);
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, keyOccurenceCount_, occupancyMask.get(), maxHashCount_);
	}
};
#endif