#pragma once

#include <cstdint>
#include <limits>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"
#include "GPUReconstruct.cuh"

#include "ErrorFlagSwapper.h"
#include "cuda_ptr.h"
#include "IGroupBy.h"
#include "AggregationFunctions.cuh"
#include "GPUTypes.h"

/// Universal null key calculator
template<typename K>
__device__ __host__ constexpr K getEmptyValue()
{
	static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
		"Unsupported data type of key (in function getEmptyValue)");

	if (std::is_integral<K>::value)
	{
		return std::numeric_limits<K>::min();
	}
	else
	{
		return std::numeric_limits<K>::quiet_NaN();
	}
}

/// Generic atomic CAS (compare and set) for any 4 and 8 bytes long data type.
template<typename T>
__device__ T genericAtomicCAS(T * address, T compare, T val)
{
	static_assert(sizeof(T) == 8 || sizeof(T) == 4, "genericAtomicCAS is working only for 4 Bytes and 8 Bytes long data types");
	if (sizeof(T) == 8)
	{
		uint64_t old = atomicCAS(reinterpret_cast<cuUInt64*>(address), *(reinterpret_cast<cuUInt64*>(&compare)), *(reinterpret_cast<cuUInt64*>(&val)));
		return *(reinterpret_cast<T*>(&old));
	}
	else if (sizeof(T) == 4)
	{
		int32_t old = atomicCAS(reinterpret_cast<int32_t*>(address), *(reinterpret_cast<int32_t*>(&compare)), *(reinterpret_cast<int32_t*>(&val)));
		return *(reinterpret_cast<T*>(&old));
	}
	else
	{
		return T{ 0 };
	}
}


/// GROUP BY Kernel processes input (inKeys and inValues). New keys from inKeys are added
/// to the hash table and values from inValues are aggregated.
/// <param name="keys">key buffer of the hash table</param>
/// <param name="values">value buffer of the hash table</param>
/// <param name="keyOccurrenceCount">key occurrences in the hash table</param>
/// <param name="maxHashCount">size of the hash table (max. number of keys)</param>
/// <param name="inKeys">input buffer with keys</param>
/// <param name="inValues">input buffer with values</param>
/// <param name="dataElementCount">count of rows in input</param>
/// <param name="errorFlag">GPU pointer to error flag</param>
template<typename AGG, typename K, typename V>
__global__ void group_by_kernel(
	K *keys,
	V *values,
	int64_t *keyOccurrenceCount,
	int32_t maxHashCount,
	K *inKeys,
	V *inValues,
	int32_t dataElementCount,
	int32_t *errorFlag) {

	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		int32_t hash = abs(static_cast<int32_t>(inKeys[i])) % maxHashCount; // TODO maybe improve hashing for float
		int32_t foundIndex = -1;

		for (int32_t j = 0; j < maxHashCount; j++) {
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
				K old = genericAtomicCAS<K>(&keys[index], getEmptyValue<K>(), inKeys[i]);

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
						printf("This will never happen: %d\n", inKeys[i]);
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
			atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_HASH_TABLE_FULL));
		}
		else
		{
			// Use aggregation of values on the bucket and the corresponding counter
			if (values)
			{
				AGG{}(&values[foundIndex], inValues[i]);
			}
			if (keyOccurrenceCount)
			{
				atomicAdd(reinterpret_cast<cuUInt64*>(&keyOccurrenceCount[foundIndex]), 1);
			}
		}
	}
}

// TODO remake to filter colConst
/// Helper kernel for mask creation from key buffer of the hash table.
template<typename K>
__global__ void is_bucket_occupied_kernel(int8_t *occupancyMask, K *keys, int32_t maxHashCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < maxHashCount; i += stride)
	{
		occupancyMask[i] = (keys[i] != getEmptyValue<K>());
		}
		}

/// GROUP BY generic class (for MIN, MAX and SUM).
template<typename AGG, typename O, typename K, typename V>
class GPUGroupBy : public IGroupBy
{
private:
static constexpr bool USE_VALUES = 
        std::is_same<AGG, AggregationFunctions::min>::value ||
        std::is_same<AGG, AggregationFunctions::max>::value ||
        std::is_same<AGG, AggregationFunctions::sum>::value ||
        std::is_same<AGG, AggregationFunctions::avg>::value;

static constexpr bool USE_KEY_OCCURRENCES = 
        std::is_same<AGG, AggregationFunctions::avg>::value ||
        std::is_same<AGG, AggregationFunctions::count>::value;

static constexpr bool DIRECT_VALUES =
        std::is_same<AGG, AggregationFunctions::min>::value ||
        std::is_same<AGG, AggregationFunctions::max>::value ||
        std::is_same<AGG, AggregationFunctions::sum>::value;

	/// Key buffer of the hash table
	K *keys_ = nullptr;
	/// Value buffer of the hash table
	V *values_ = nullptr;
	/// Count of values aggregated per key (helper buffer of the hash table)
	int64_t *keyOccurrenceCount_ = nullptr;

	/// Size of the hash table (max. count of unique keys)
	const int32_t maxHashCount_;
	/// Error flag swapper for error checking after kernel runs
	ErrorFlagSwapper errorFlagSwapper_;

public:
	/// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurrence counts)
	/// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
	GPUGroupBy(int32_t maxHashCount) :
		maxHashCount_(maxHashCount)
	{
		try
		{
			GPUMemory::alloc(&keys_, maxHashCount_);
            if (USE_VALUES)
            {
				GPUMemory::alloc(&values_, maxHashCount_);
			}
            if (USE_KEY_OCCURRENCES)
            {
				GPUMemory::allocAndSet(&keyOccurrenceCount_, 0, maxHashCount_);
			}
		}
		catch(...)
		{
			if(keys_)
			{
				GPUMemory::free(keys_);
			}
			if(values_)
			{
				GPUMemory::free(values_);
			}
			if(keyOccurrenceCount_)
			{
				GPUMemory::free(keyOccurrenceCount_);
			}
			throw;
		}
		GPUMemory::fillArray(keys_, getEmptyValue<K>(), maxHashCount_);
        if (USE_VALUES)
        {
			GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
		}
	}

	/// Create GPUGroupBy object with existing keys (allocate whole new hash table)
	/// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
	/// <param name="keys">GPU buffer with existing keys (will be copied to a new buffer)</param>
	GPUGroupBy(int32_t maxHashCount, K * keys) :
		GPUGroupBy(maxHashCount)
	{
		GPUMemory::copyDeviceToDevice(keys_, keys, maxHashCount_);
	}

	~GPUGroupBy()
	{
		GPUMemory::free(keys_);
		if (USE_VALUES)
		{
			GPUMemory::free(values_);
		}
		if (USE_KEY_OCCURRENCES)
		{
			GPUMemory::free(keyOccurrenceCount_);
		}
	}

	GPUGroupBy(const GPUGroupBy &) = delete;
	GPUGroupBy& operator=(const GPUGroupBy &) = delete;

	/// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
	/// <param name="inKeys">input buffer with keys</param>
	/// <param name="inValues">input buffer with values</param>
	/// <param name="dataElementCount">row count to process</param>
	void groupBy(K *inKeys, V *inValues, int32_t dataElementCount)
	{
		if (dataElementCount > 0)
		{
			group_by_kernel <AGG> << <  Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
				(keys_, values_, keyOccurrenceCount_, maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.GetFlagPointer());
			errorFlagSwapper_.Swap();
		}
	}

	/// Get the size of the hash table (max. count of unique keys)
	/// <returns>size of the hash table</returns>
	int32_t getMaxHashCount()
	{
		return maxHashCount_;
	}

	/// Reconstruct needed raw fields (do not calculate final results yet)
	/// Reconstruct keys, values and key occurrence counts separately
	/// <param name="keys">output buffer to fill with reconstructed keys</param>
	/// <param name="values">output buffer to fill with reconstructed values</param>
	/// <param name="occurrences">not used buffer if using operations MIN, MAX or SUM - nullptr can be used</param>
	/// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
	void reconstructRawNumbers(K * keys, V * values, int64_t * occurrences, int32_t * elementCount)
	{
		cuda_ptr<int8_t> occupancyMask(maxHashCount_, 0);
		is_bucket_occupied_kernel << < Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask.get(), keys_, maxHashCount_);
		GPUReconstruct::reconstructCol(keys, elementCount, keys_, occupancyMask.get(), maxHashCount_);
		if (USE_VALUES)
		{
			GPUReconstruct::reconstructCol(values, elementCount, values_, occupancyMask.get(), maxHashCount_);
		}
		if (USE_KEY_OCCURRENCES)
		{
			GPUReconstruct::reconstructCol(occurrences, elementCount, keyOccurrenceCount_, occupancyMask.get(), maxHashCount_);
		}
	}

	/// Get the final results of GROUP BY operation - for operations Min, Max and Sum on single GPU
	/// <param name="outKeys">double pointer of output GPU buffer (will be allocated and filled with final keys)</param>
	/// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
	/// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
	void getResults(K **outKeys, O **outValues, int32_t *outDataElementCount)
	{
		static_assert(std::is_integral<K>::value || std::is_floating_point<K>::value,
			"GPUGroupBy<min/max/sum>.getResults K (keys) must be integral or floating point");
		static_assert(std::is_integral<V>::value || std::is_floating_point<V>::value,
			"GPUGroupBy<min/max/sum>.getResults V (values) must be integral or floating point");

		// Create buffer for bucket compression - reconstruct
		cuda_ptr<int8_t> occupancyMask(maxHashCount_, 0);

		// Calculate occupancy mask
		is_bucket_occupied_kernel << < Context::getInstance().calcGridDim(maxHashCount_), Context::getInstance().getBlockDim() >> >
			(occupancyMask.get(), keys_, maxHashCount_);

		// Reconstruct the keys
		GPUReconstruct::reconstructColKeep(outKeys, outDataElementCount, keys_, occupancyMask.get(), maxHashCount_);

		// Reconstruct aggregated values according to the operation
        if (DIRECT_VALUES)  // for min, max and sum: values_ are direct results, just reconstruct them
        {
            if (!std::is_same<O, V>::value)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR, "Input and output value data type must be the same in GROUP BY");
            }
            // reinterpret_cast is needed to solve compilation error
			GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, reinterpret_cast<O*>(values_),
					occupancyMask.get(), maxHashCount_);
		}
		
        else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg: values_ need to be divided by keyOccurrences_ and reconstructed
        {
			cuda_ptr<O> outValuesGPU(maxHashCount_);
            // Divide by counts to get averages for buckets
            try
            {
                GPUArithmetic::colCol<ArithmeticOperations::div>(outValuesGPU.get(), values_, keyOccurrenceCount_, maxHashCount_);
            }
            catch (query_engine_error& err)
            {
                // Rethrow just if error is not division by zero.
                // Division by zero is OK here because it is more efficient to perform division
                // on raw (not reconstructed) hash table - and some keyOccurrences here can be 0.
                if (err.GetQueryEngineError() != QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR)
                {
                    throw err; 
                }
            }
            // Reonstruct result with original occupancyMask
            GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, outValuesGPU.get(), occupancyMask.get(), maxHashCount_);
		}
        else // for count: reconstruct and return keyOccurrences_
        {
            // reinterpret_cast is needed to solve compilation error
            // not reinterpreting anything here actually, outValues is int64_t** always in this else-branch
            GPUReconstruct::reconstructColKeep(reinterpret_cast<int64_t**>(outValues), outDataElementCount, keyOccurrenceCount_,
                                               occupancyMask.get(), maxHashCount_);
		}
	}
	
	/// Merge results from all devices and store to buffers on default device (multi GPU function)
	/// <param name="outKeys">double pointer of output GPU buffer (will be allocated and filled with final keys)</param>
	/// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
	/// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
	/// <param name="tables">vector of unique pointers to IGroupBy objects with hash tables on every device (GPU)</param>
	void getResults(K** outKeys,
					O** outValues,
					int32_t* outDataElementCount,
					std::vector<std::unique_ptr<IGroupBy>>& tables)
	{
		if (tables.size() <= 0) // invalid count of tables
		{
			throw std::invalid_argument("Number of tables have to be at least 1.");
		}
		else if (tables.size() == 1 || tables[1].get() == nullptr) // just one table
		{
			getResults(outKeys, outValues, outDataElementCount);
		}
		else // more tables
		{
			int oldDeviceId = Context::getInstance().getBoundDeviceID();

			std::vector<K> keysAllHost;
			std::vector<V> valuesAllHost;
			std::vector<int64_t> occurrencesAllHost;
			int32_t sumElementCount = 0;

			// Collect data from all devices (graphic cards) to host
			for (int i = 0; i < tables.size(); i++)
			{
				if (tables[i].get() == nullptr)
				{
					break;
				}
				// TODO change to cudaMemcpyPeerAsync
				GPUGroupBy<AGG, O, K, V>* table = reinterpret_cast<GPUGroupBy<AGG, O, K, V>*>(tables[i].get());
				std::unique_ptr<K[]> keys = std::make_unique<K[]>(table->getMaxHashCount());
				std::unique_ptr<V[]> values = std::make_unique<V[]>(table->getMaxHashCount());
				std::unique_ptr<int64_t[]> occurrences = std::make_unique<int64_t[]>(table->getMaxHashCount());
				int32_t elementCount;
				Context::getInstance().bindDeviceToContext(i);

				// Reconstruct raw keys, values (conditional) and occurrences (conditional)
				table->reconstructRawNumbers(keys.get(), values.get(), occurrences.get(), &elementCount);

				// Append data to host vectors
				keysAllHost.insert(keysAllHost.end(), keys.get(), keys.get() + elementCount);
				if (USE_VALUES)
				{
					valuesAllHost.insert(valuesAllHost.end(), values.get(), values.get() + elementCount);
				}
				if (USE_KEY_OCCURRENCES)
				{
					occurrencesAllHost.insert(occurrencesAllHost.end(), occurrences.get(), occurrences.get() + elementCount);
				}
				sumElementCount += elementCount;
			}

			Context::getInstance().bindDeviceToContext(oldDeviceId);
			if (sumElementCount > 0)
			{
				cuda_ptr<K> keysAllGPU(sumElementCount);
				cuda_ptr<V> valuesAllGPU(sumElementCount);
				cuda_ptr<int64_t> occurrencesAllGPU(sumElementCount);

				// Copy the condens from host to default device
				GPUMemory::copyHostToDevice(keysAllGPU.get(), keysAllHost.data(), sumElementCount);
				if (USE_VALUES)
				{
					GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);
				}
				if (USE_KEY_OCCURRENCES)
				{
					GPUMemory::copyHostToDevice(occurrencesAllGPU.get(), occurrencesAllHost.data(), sumElementCount);
				}

				// Merge results
				if (DIRECT_VALUES)	// for min, max and sum
				{
					GPUGroupBy<AGG, O, K, V> finalGroupBy(sumElementCount);
					finalGroupBy.groupBy(keysAllGPU.get(), valuesAllGPU.get(), sumElementCount);
					finalGroupBy.getResults(outKeys, outValues, outDataElementCount);
				}
				else if (std::is_same<AGG, AggregationFunctions::avg>::value)	// for avg
				{
					V* valuesMerged = nullptr;
					int64_t* occurrencesMerged = nullptr;

					// Calculate sum of values
					// Initialize new empty sumGroupBy table
					K* tempKeys = nullptr;
					GPUGroupBy<AggregationFunctions::sum, V, K, V> sumGroupBy(sumElementCount);
					sumGroupBy.groupBy(keysAllGPU.get(), valuesAllGPU.get(), sumElementCount);
					sumGroupBy.getResults(&tempKeys, &valuesMerged, outDataElementCount);

					// Calculate sum of occurrences
					// Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
					GPUGroupBy<AggregationFunctions::sum, int64_t, K, int64_t> countGroupBy(*outDataElementCount, tempKeys);
					countGroupBy.groupBy(keysAllGPU.get(), occurrencesAllGPU.get(), sumElementCount);
					countGroupBy.getResults(outKeys, &occurrencesMerged, outDataElementCount);
					GPUMemory::alloc(outValues, *outDataElementCount);
					GPUArithmetic::colCol<ArithmeticOperations::div>(*outValues, valuesMerged, occurrencesMerged, *outDataElementCount);
					GPUMemory::free(valuesMerged);
					GPUMemory::free(occurrencesMerged);
					GPUMemory::free(tempKeys);
				}
				else	// for count
				{
					if (!std::is_same<O, int64_t>::value)
					{
						CheckQueryEngineError(GPU_EXTENSION_ERROR, "Output value data type in GROUP BY with COUNT must be int64_t");
					}
					GPUGroupBy<AggregationFunctions::sum, int64_t, K, int64_t> finalGroupBy(sumElementCount);
					finalGroupBy.groupBy(keysAllGPU.get(), occurrencesAllGPU.get(), sumElementCount);
												  // reinterpret_cast is needed to solve compilation error
					finalGroupBy.getResults(outKeys, reinterpret_cast<int64_t**>(outValues), outDataElementCount);
				}
			}
			else
			{
				*outDataElementCount = 0;
			}
		}
	}
};


// Specializations for String keys
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::string, V>;

template <typename O, typename V>
class GPUGroupBy<AggregationFunctions::avg, O, std::string, V>;	// TODO delete specs.

template <typename V>
class GPUGroupBy<AggregationFunctions::count, int64_t, std::string, V>;	// TODO delete specs.

// Specializations for multi-keys (GROUP BY multiple column)
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::vector<void*>, V>;
