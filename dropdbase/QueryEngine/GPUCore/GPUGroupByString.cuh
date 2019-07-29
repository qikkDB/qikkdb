#pragma once

#include "GPUFilter.cuh"
#include "GPUFilterConditions.cuh"
#include "GPUGroupBy.cuh"
#include "GPUStringUnary.cuh"


constexpr int32_t GBS_SOURCE_INDEX_EMPTY_KEY = -1;
constexpr int32_t GBS_SOURCE_INDEX_KEY_IN_BUFFER = -2;
constexpr int32_t GBS_STRING_HASH_SEED = 31;


__device__ int32_t GetHash(char* text, int32_t length);


__device__ bool AreEqualStrings(char* textA, int32_t lenghtA, GPUMemory::GPUString stringColB, int64_t indexB);


__device__ bool IsNewKey(char* checkedKeyChars,
                         int32_t checkedKeyLength,
                         GPUMemory::GPUString inKeys,
                         GPUMemory::GPUString keysBuffer,
                         int32_t* sourceIndices,
                         int32_t index);


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
template <typename AGG, typename V>
__global__ void kernel_group_by_string(int32_t* sourceIndices,
                                       int32_t* stringLengths,
                                       GPUMemory::GPUString keysBuffer,
                                       V* values,
                                       int64_t* keyOccurrenceCount,
                                       int32_t maxHashCount,
                                       GPUMemory::GPUString inKeys,
                                       V* inValues,
                                       int32_t dataElementCount,
                                       int32_t* errorFlag)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int64_t inKeyIndex = GetStringIndex(inKeys.stringIndices, i);
        const int32_t inKeyLength = GetStringLength(inKeys.stringIndices, i);
        char* inKeyChars = inKeys.allChars + inKeyIndex;
        // Calculate hash
        const int32_t hash = abs(GetHash(inKeyChars, inKeyLength)) % maxHashCount;

        int32_t foundIndex = -1;
        for (int32_t j = 0; j < maxHashCount; j++)
        {
            // Calculate index to hash-table from hash
            const int32_t index = (hash + j) % maxHashCount;

            // Check if key is not empty and key is not equal to the currently inserted key
            if (sourceIndices[index] != GBS_SOURCE_INDEX_EMPTY_KEY &&
                IsNewKey(inKeyChars, inKeyLength, inKeys, keysBuffer, sourceIndices, index))
            {
                continue;
            }

            // If key is empty
            if (sourceIndices[index] == GBS_SOURCE_INDEX_EMPTY_KEY)
            {
                // Compare key at index with Empty and if equals, store there inKey
                int32_t old =
                    genericAtomicCAS<int32_t>(&sourceIndices[index], GBS_SOURCE_INDEX_EMPTY_KEY, i);

                // Check if some other thread stored different key to this index
                if (old != GBS_SOURCE_INDEX_EMPTY_KEY && old != i &&
                    IsNewKey(inKeyChars, inKeyLength, inKeys, keysBuffer, sourceIndices, index))
                {
                    continue; // Try to find another index
                }
            }
            else if (sourceIndices[index] != i &&
                     IsNewKey(inKeyChars, inKeyLength, inKeys, keysBuffer, sourceIndices, index))
            {
                continue; // try to find another index
            }

            // The key was added or found as already existing
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
            stringLengths[foundIndex] = inKeyLength;
            // Aggregate value
            if (values)
            {
                AGG{}(&values[foundIndex], inValues[i]);
            }
            // Increment occurrence counter
            if (keyOccurrenceCount)
            {
                atomicAdd(reinterpret_cast<cuUInt64*>(&keyOccurrenceCount[foundIndex]), 1);
            }
        }
    }
}


__global__ void kernel_collect_string_keys(GPUMemory::GPUString sideBuffer,
                                           int32_t* sourceIndices,
                                           int32_t* stringLengths,
                                           GPUMemory::GPUString keysBuffer,
                                           int32_t maxHashCount,
                                           GPUMemory::GPUString inKeys,
                                           int32_t inKeysCount);


__global__ void kernel_source_indices_to_mask(int8_t* occupancyMask, int32_t* sourceIndices, int32_t maxHashCount);


__global__ void kernel_mark_collected_strings(int32_t* sourceIndices, int32_t maxHashCount);


/// GROUP BY generic class for String keys
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::string, V> : public IGroupBy
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

public:
    /// Temp buffer where one value points to input key
    /// or tells the key is already in keysBuffer_
    int32_t* sourceIndices_ = nullptr;
    /// Buffer with lengths of collected string keys
    int32_t* stringLengths_ = nullptr;
    /// Buffer with collected string keys
    GPUMemory::GPUString keysBuffer_ {nullptr, nullptr};

private:
    /// Value buffer of the hash table
    V* values_ = nullptr;
	int8_t* valuesNullMask_ = nullptr;

    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurrenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    const int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurrence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount) : maxHashCount_(maxHashCount)
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::allocAndSet(&stringLengths_, 0, maxHashCount_);
            // And for values and occurrences
            if (USE_VALUES)
            {
				GPUMemory::alloc(&values_, maxHashCount_);
				GPUMemory::allocAndSet(&valuesNullMask_, 1, maxHashCount_);
			}
            if (USE_KEY_OCCURRENCES)
            {
				GPUMemory::allocAndSet(&keyOccurrenceCount_, 0, maxHashCount_);
			}
        }
        catch (...)
        {
            if (sourceIndices_)
            {
                GPUMemory::free(sourceIndices_);
            }
            if (stringLengths_)
            {
                GPUMemory::free(stringLengths_);
            }
            if (values_)
            {
                GPUMemory::free(values_);
            }
			if(valuesNullMask_)
			{
				GPUMemory::free(valuesNullMask_);
			}
            if (keyOccurrenceCount_)
            {
                GPUMemory::free(keyOccurrenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        if (USE_VALUES)
        {
            GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
        }
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="sourceIndices">GPU buffer with original source indices (will be copied to a new buffer)</param>
    /// <param name="stringLengths">GPU buffer with lengths of original string keys (will be copied to a new buffer)</param>
    /// <param name="keysBuffer">GPU buffer with original existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, int32_t* stringLengths, GPUMemory::GPUString keysBuffer) :
		GPUGroupBy(maxHashCount)
    {
        int64_t totalCharCount = 0;
        try
        {
            // Alloc string key buffer
            GPUMemory::alloc(&(keysBuffer_.stringIndices), maxHashCount_);
            GPUMemory::copyDeviceToHost(&totalCharCount, keysBuffer.stringIndices + maxHashCount_ - 1, 1);
            if (totalCharCount > 0)
            {
                GPUMemory::alloc(&(keysBuffer_.allChars), totalCharCount);
            }
            else
            {
                keysBuffer_.allChars = nullptr;
            }
        }
        catch (...)
        {
            GPUMemory::free(keysBuffer_);
            throw;
        }
        // Copy string keys
        GPUMemory::copyDeviceToDevice(sourceIndices_, sourceIndices, maxHashCount_);
        GPUMemory::copyDeviceToDevice(stringLengths_, stringLengths, maxHashCount_);
        GPUMemory::copyDeviceToDevice(keysBuffer_.stringIndices, keysBuffer.stringIndices, maxHashCount_);
        if (totalCharCount > 0)
        {
            GPUMemory::copyDeviceToDevice(keysBuffer_.allChars, keysBuffer.allChars, totalCharCount);
        }
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        GPUMemory::free(stringLengths_);
        GPUMemory::free(keysBuffer_);

        if (USE_VALUES)
        {
            GPUMemory::free(values_);
            GPUMemory::free(valuesNullMask_);
        }
        if (USE_KEY_OCCURRENCES)
        {
            GPUMemory::free(keyOccurrenceCount_);
        }
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffer with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void ProcessBlock(GPUMemory::GPUString inKeys, V* inValues, int32_t dataElementCount)
    {
        if (dataElementCount > 0)
        {
            Context& context = Context::getInstance();
            kernel_group_by_string<AGG><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                sourceIndices_, stringLengths_, keysBuffer_, values_, keyOccurrenceCount_,
                maxHashCount_, inKeys, inValues, dataElementCount, errorFlagSwapper_.GetFlagPointer());
            errorFlagSwapper_.Swap();

            GPUMemory::GPUString sideBuffer;
            try
            {
                GPUMemory::alloc(&(sideBuffer.stringIndices), maxHashCount_);

                GPUReconstruct::PrefixSum(sideBuffer.stringIndices, stringLengths_, maxHashCount_);

                int64_t totalCharCount;
                GPUMemory::copyDeviceToHost(&totalCharCount, sideBuffer.stringIndices + maxHashCount_ - 1, 1);
                GPUMemory::alloc(&(sideBuffer.allChars), totalCharCount);

                kernel_collect_string_keys<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                    sideBuffer, sourceIndices_, stringLengths_, keysBuffer_, maxHashCount_, inKeys, dataElementCount);

                GPUMemory::free(keysBuffer_);
                keysBuffer_ = sideBuffer;

                kernel_mark_collected_strings<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                    sourceIndices_, maxHashCount_);
            }
            catch (...)
            {
                GPUMemory::free(sideBuffer);
                throw;
            }
            CheckCudaError(cudaGetLastError());
        }
    }


    /// Get the size of the hash table (max. count of unique keys)
    /// <returns>size of the hash table</returns>
    int32_t GetMaxHashCount()
    {
        return maxHashCount_;
    }


    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurrence counts separately
    /// <param name="keys">output buffer to fill with reconstructed keys</param>
    /// <param name="values">output buffer to fill with reconstructed values</param>
    /// <param name="occurrences">not used buffer if using operations MIN, MAX or SUM - nullptr can be used</param>
    /// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void ReconstructRawNumbers(std::vector<int32_t>& keysStringLengths,
                               std::vector<char>& keysAllChars,
                               V* values,
                               int64_t* occurrences,
                               int32_t* elementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        GPUReconstruct::ReconstructStringColRaw(keysStringLengths, keysAllChars, elementCount,
                                                keysBuffer_, occupancyMask.get(), maxHashCount_);
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
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
    void GetResults(GPUMemory::GPUString* outKeys, O** outValues, int32_t* outDataElementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);
        GPUReconstruct::ReconstructStringColKeep(outKeys, outDataElementCount, keysBuffer_,
                                                 occupancyMask.get(), maxHashCount_);
        
        // Reconstruct aggregated values
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
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final
    /// keys)</param> <param name="outValues">double pointer of output GPU buffer (will be allocated
    /// and filled with final values)</param> <param name="outDataElementCount">output CPU buffer
    /// (will be filled with count of reconstructed elements)</param> <param name="tables">vector of
    /// unique pointers to IGroupBy objects with hash tables on every device (GPU)</param>
    void GetResults(GPUMemory::GPUString* outKeys,
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
            GetResults(outKeys, outValues, outDataElementCount);
        }
        else // more tables
        {
            int oldDeviceId = Context::getInstance().getBoundDeviceID();

            std::vector<int32_t> keysAllHostStringLengths;
            std::vector<char> keysAllHostAllChars;
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
                GPUGroupBy<AGG, O, std::string,V>* table =
                    reinterpret_cast<GPUGroupBy<AGG, O, std::string,V>*>(tables[i].get());
                std::vector<int32_t> keysStringLengths;
                std::vector<char> keysAllChars;
                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->GetMaxHashCount());
                std::unique_ptr<int64_t[]> occurrences = std::make_unique<int64_t[]>(table->GetMaxHashCount());
                int32_t elementCount;
                Context::getInstance().bindDeviceToContext(i);

                // Reconstruct keys and values
                table->ReconstructRawNumbers(keysStringLengths, keysAllChars, values.get(), occurrences.get(), &elementCount);

                // Append data to host vectors
                keysAllHostStringLengths.insert(keysAllHostStringLengths.end(),
                                                keysStringLengths.begin(), keysStringLengths.end());
                keysAllHostAllChars.insert(keysAllHostAllChars.end(), keysAllChars.begin(),
                                           keysAllChars.end());
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
                cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
                cuda_ptr<V> valuesAllGPU(sumElementCount);
				cuda_ptr<int64_t> occurrencesAllGPU(sumElementCount);

                // Copy the condens from host to default device
                GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                            keysAllHostStringLengths.data(), sumElementCount);
                if (USE_VALUES)
                {
                    GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);
                }
                if (USE_KEY_OCCURRENCES)
                {
                    GPUMemory::copyHostToDevice(occurrencesAllGPU.get(), occurrencesAllHost.data(), sumElementCount);
                }

                // Construct new GPUString
                GPUMemory::GPUString keysAllGPU;
                GPUMemory::alloc(&(keysAllGPU.stringIndices), sumElementCount);
                GPUReconstruct::PrefixSum(keysAllGPU.stringIndices, keysAllGPUStringLengths.get(), sumElementCount);
                GPUMemory::alloc(&(keysAllGPU.allChars), keysAllHostAllChars.size());
                GPUMemory::copyHostToDevice(keysAllGPU.allChars, keysAllHostAllChars.data(),
                                            keysAllHostAllChars.size());

                // Merge results
                if (DIRECT_VALUES) // for min, max and sum
                {
                    GPUGroupBy<AGG, O, std::string,V> finalGroupBy(sumElementCount);
                    finalGroupBy.ProcessBlock(keysAllGPU, valuesAllGPU.get(), sumElementCount);
                    finalGroupBy.GetResults(outKeys, outValues, outDataElementCount);
                }
                else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg
                {
                    V* valuesMerged = nullptr;
                    int64_t* occurrencesMerged = nullptr;

                    // Calculate sum of values
                    // Initialize new empty sumGroupBy table
                    GPUMemory::GPUString keysToDiscard;
                    GPUGroupBy<AggregationFunctions::sum, V, std::string, V> sumGroupBy(sumElementCount);
                    sumGroupBy.ProcessBlock(keysAllGPU, valuesAllGPU.get(), sumElementCount);
                    sumGroupBy.GetResults(&keysToDiscard, &valuesMerged, outDataElementCount);
                    GPUMemory::free(keysToDiscard);

                    // Calculate sum of occurrences
                    // Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> countGroupBy(sumElementCount, sumGroupBy.sourceIndices_, sumGroupBy.stringLengths_, sumGroupBy.keysBuffer_);
                    countGroupBy.ProcessBlock(keysAllGPU, occurrencesAllGPU.get(), sumElementCount);
                    countGroupBy.GetResults(outKeys, &occurrencesMerged, outDataElementCount);

                    // Divide merged values by merged occurrences to get final averages
                    GPUMemory::alloc(outValues, *outDataElementCount);
                    GPUArithmetic::colCol<ArithmeticOperations::div>(*outValues, valuesMerged, occurrencesMerged, *outDataElementCount);
                    
                    GPUMemory::free(valuesMerged);
                    GPUMemory::free(occurrencesMerged);
                }
                else // for count
                {
					if (!std::is_same<O, int64_t>::value)
					{
						CheckQueryEngineError(GPU_EXTENSION_ERROR, "Output value data type in GROUP BY with COUNT must be int64_t");
					}
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> finalGroupBy(sumElementCount);
                    finalGroupBy.ProcessBlock(keysAllGPU, occurrencesAllGPU.get(), sumElementCount);
												  // reinterpret_cast is needed to solve compilation error
                    finalGroupBy.GetResults(outKeys, reinterpret_cast<int64_t**>(outValues), outDataElementCount);
                }

                GPUMemory::free(keysAllGPU);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};
