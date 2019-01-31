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

// Universal null key calculator
template<typename T>
__device__ __host__ constexpr T getEmptyValue()
{
	static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
		"Unsupported data type in group by agg function template");

	if (std::is_integral<T>::value)
	{
		return std::numeric_limits<T>::min();
	}
	else if (std::is_floating_point<T>::value)
	{
		return std::numeric_limits<T>::quiet_NaN();
	}
}

// Kernel
template<typename AGG, typename K, typename V>
__global__ void group_by_kernel(
	K *keys,
	V *values,
	int32_t *keyOccurenceCount,
	int32_t maxHashCount,
	K *inKeys,
	V *inValues,
	int32_t dataElementCount,
	int32_t *errorFlag) {

	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		int32_t hash = abs(static_cast<int32_t>(inKeys[i]));
		int32_t foundIndex = -1;
		for (int j = 0; j < maxHashCount; j++) {
			// Calculate hash - use type conversion because of float
			int32_t index = (hash + j) % maxHashCount;

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
			atomicAdd(&keyOccurenceCount[foundIndex], 1);
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

template<typename AGG, typename K, typename V>
class GPUGroupBy
{
private:
	K *keys_;						// Keys
	V *values_;						// Values
	int32_t *keyOccurenceCount_;	// Count of occurrances of keys		

	int32_t maxHashCount_;			// Maximum size of the result hash table

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
	void getResults(K *outKeys, V *outValues, int32_t *outDataElementCount)
	{
		static_assert((std::is_integral<K>::value || std::is_floating_point<K>::value) &&
			(std::is_integral<V>::value || std::is_floating_point<V>::value),
			"Unsupported data type in group by get results function template");

		// Create buffer for bucket compression - reconstruct
		int32_t *occupancyMask;
		GPUMemory::allocAndSet(&occupancyMask, 0, maxHashCount_);

		// Calculate fill mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask, keys_, maxHashCount_);

		// Reconstruct the output
		// Copy back the results based on the operation
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask, maxHashCount_);
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_, occupancyMask, maxHashCount_);

		GPUMemory::free(occupancyMask);
	}
};

template<typename K, typename V>
class GPUGroupBy<AggregationFunctions::avg, K, V>
{
private:
	K *keys_;						// Keys
	V *values_;						// Values
	int32_t *keyOccurenceCount_;	// Count of occurrances of keys		

	int32_t maxHashCount_;			// Maximum size of the result hash table

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
	void getResults(K *outKeys, V *outValues, int32_t *outDataElementCount)
	{
		static_assert((std::is_integral<K>::value || std::is_floating_point<K>::value) &&
			(std::is_integral<V>::value || std::is_floating_point<V>::value),
			"Unsupported data type in group by get results function template");

		// Create buffer for bucket compression - reconstruct
		int32_t *occupancyMask;
		GPUMemory::allocAndSet(&occupancyMask, 0, maxHashCount_);

		// Calculate fill mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask, keys_, maxHashCount_);

		// Reconstruct the output
		// Divide by count to get average for buckets
		GPUArithmetic::division(values_, values_, keyOccurenceCount_, maxHashCount_);

		// TODO if V is integral outvalues should be float
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask, maxHashCount_);
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_, occupancyMask, maxHashCount_);

		GPUMemory::free(occupancyMask);
	}
};

template<typename K, typename V>
class GPUGroupBy<AggregationFunctions::cnt, K, V>
{
private:
	K *keys_;						// Keys
	V *values_;						// Values
	int32_t *keyOccurenceCount_;	// Count of occurrances of keys		

	int32_t maxHashCount_;			// Maximum size of the result hash table

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
	void getResults(K *outKeys, V *outValues, int32_t *outDataElementCount)
	{
		static_assert((std::is_integral<K>::value || std::is_floating_point<K>::value) &&
			(std::is_integral<V>::value || std::is_floating_point<V>::value),
			"Unsupported data type in group by get results function template");

		// Create buffer for bucket compression - reconstruct
		int32_t *occupancyMask;
		GPUMemory::allocAndSet(&occupancyMask, 0, maxHashCount_);

		// Calculate fill mask
		is_bucket_occupied_kernel << <  Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask, keys_, maxHashCount_);

		// Reconstruct the output
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask, maxHashCount_);
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, keyOccurenceCount_, occupancyMask, maxHashCount_);

		GPUMemory::free(occupancyMask);
	}
};
#endif