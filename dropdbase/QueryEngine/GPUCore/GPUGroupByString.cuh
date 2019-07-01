#pragma once

#include "GPUFilter.cuh"
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
/// <param name="keyOccurenceCount">key occurences in the hash table</param>
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
                                       int64_t* keyOccurenceCount,
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
        const int32_t hash = GetHash(inKeyChars, inKeyLength);

        int32_t foundIndex = -1;
        for (int32_t j = 0; j < maxHashCount; j++)
        {
            // Calculate index to hash-table from hash
            const int32_t index = abs((hash + j) % maxHashCount);

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
            // Use aggregation of values on the bucket and the corresponding counter
            AGG{}(&values[foundIndex], inValues[i]);
            atomicAdd(reinterpret_cast<cuUInt64*>(&keyOccurenceCount[foundIndex]), 1);
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


__global__ void kernel_is_bucket_occupied(int8_t* occupancyMask, int32_t* sourceIndices, int32_t maxHashCount);


__global__ void kernel_mark_collected_strings(int32_t* sourceIndices, int32_t maxHashCount);


/// GROUP BY String (semi-generic class) for MIN, MAX and SUM.
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::string, V> : public IGroupBy
{
public:
    int32_t* sourceIndices_ = nullptr;
    int32_t* stringLengths_ = nullptr;
    GPUMemory::GPUString keysBuffer_ {nullptr, nullptr};

private:
    /// Value buffer of the hash table
    V* values_ = nullptr;
    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount) : maxHashCount_(maxHashCount)
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::allocAndSet(&stringLengths_, 0, maxHashCount_);
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="keys">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, int32_t* stringLengths, GPUMemory::GPUString keysBuffer) : maxHashCount_(maxHashCount)
    {
        int64_t totalCharCount = 0;
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&stringLengths_, maxHashCount_);
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
            
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            GPUMemory::free(keysBuffer_);
            throw;
        }
        GPUMemory::copyDeviceToDevice(sourceIndices_, sourceIndices, maxHashCount_);
        GPUMemory::copyDeviceToDevice(stringLengths_, stringLengths, maxHashCount_);
        GPUMemory::copyDeviceToDevice(keysBuffer_.stringIndices, keysBuffer.stringIndices, maxHashCount_);
        if (totalCharCount > 0)
        {
            GPUMemory::copyDeviceToDevice(keysBuffer_.allChars, keysBuffer.allChars, totalCharCount);
        }
        GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        GPUMemory::free(stringLengths_);
        GPUMemory::free(keysBuffer_);

        GPUMemory::free(values_);
        GPUMemory::free(keyOccurenceCount_);
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffer with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void groupBy(GPUMemory::GPUString inKeys, V* inValues, int32_t dataElementCount)
    {
        if (dataElementCount > 0)
        {
            Context& context = Context::getInstance();
            kernel_group_by_string<AGG><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                sourceIndices_, stringLengths_, keysBuffer_, values_, keyOccurenceCount_,
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
    int32_t getMaxHashCount()
    {
        return maxHashCount_;
    }


    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurence counts separately
    /// <param name="keys">output buffer to fill with reconstructed keys</param>
    /// <param name="values">output buffer to fill with reconstructed values</param>
    /// <param name="occurences">not used buffer if using operations MIN, MAX or SUM - nullptr can be used</param>
    /// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void reconstructRawNumbers(std::vector<int32_t>& keysStringLengths,
                               std::vector<char>& keysAllChars,
                               V* values,
                               int64_t* occurences,
                               int32_t* elementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        GPUReconstruct::ReconstructStringColRaw(keysStringLengths, keysAllChars, elementCount,
                                                keysBuffer_, occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructCol(values, elementCount, values_, occupancyMask.get(), maxHashCount_);
    }


    /// Get the final results of GROUP BY operation - for operations Min, Max and Sum on single GPU
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
    void getResults(GPUMemory::GPUString* outKeys, O** outValues, int32_t* outDataElementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);
        GPUReconstruct::ReconstructStringColKeep(outKeys, outDataElementCount, keysBuffer_,
                                                 occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_,
                                           occupancyMask.get(), maxHashCount_);
    }


    /// Merge results from all devices and store to buffers on default device (multi GPU function)
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final
    /// keys)</param> <param name="outValues">double pointer of output GPU buffer (will be allocated
    /// and filled with final values)</param> <param name="outDataElementCount">output CPU buffer
    /// (will be filled with count of reconstructed elements)</param> <param name="tables">vector of
    /// unique pointers to IGroupBy objects with hash tables on every device (GPU)</param>
    void getResults(GPUMemory::GPUString* outKeys,
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

            std::vector<int32_t> keysAllHostStringLengths;
            std::vector<char> keysAllHostAllChars;
            std::vector<V> valuesAllHost;
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
                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->getMaxHashCount());
                int32_t elementCount;
                Context::getInstance().bindDeviceToContext(i);

                // Reconstruct keys and values
                table->reconstructRawNumbers(keysStringLengths, keysAllChars, values.get(), nullptr, &elementCount);

                // Append data to host vectors
                keysAllHostStringLengths.insert(keysAllHostStringLengths.end(),
                                                keysStringLengths.begin(), keysStringLengths.end());
                keysAllHostAllChars.insert(keysAllHostAllChars.end(), keysAllChars.begin(),
                                           keysAllChars.end());
                valuesAllHost.insert(valuesAllHost.end(), values.get(), values.get() + elementCount);
                sumElementCount += elementCount;
            }

            Context::getInstance().bindDeviceToContext(oldDeviceId);
            if (sumElementCount > 0)
            {
                cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
                cuda_ptr<V> valuesAllGPU(sumElementCount);

                // Copy the condens from host to default device
                GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                            keysAllHostStringLengths.data(), sumElementCount);
                GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);

                // Construct new GPUString
                GPUMemory::GPUString keysAllGPU;
                GPUMemory::alloc(&(keysAllGPU.stringIndices), sumElementCount);
                GPUReconstruct::PrefixSum(keysAllGPU.stringIndices, keysAllGPUStringLengths.get(), sumElementCount);
                GPUMemory::alloc(&(keysAllGPU.allChars), keysAllHostAllChars.size());
                GPUMemory::copyHostToDevice(keysAllGPU.allChars, keysAllHostAllChars.data(),
                                            keysAllHostAllChars.size());

                // Merge results
                GPUGroupBy<AGG, O, std::string,V> finalGroupBy(sumElementCount);
                finalGroupBy.groupBy(keysAllGPU, valuesAllGPU.get(), sumElementCount);
                finalGroupBy.getResults(outKeys, outValues, outDataElementCount);

                GPUMemory::free(keysAllGPU);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};



/// GROUP BY String, specialization for AVG 
template <typename O, typename V>
class GPUGroupBy<AggregationFunctions::avg, O, std::string, V> : public IGroupBy
{
private:
    int32_t* sourceIndices_ = nullptr;
    int32_t* stringLengths_ = nullptr;
    GPUMemory::GPUString keysBuffer_ {nullptr, nullptr};

    /// Value buffer of the hash table
    V* values_ = nullptr;
    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount) : maxHashCount_(maxHashCount)
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::allocAndSet(&stringLengths_, 0, maxHashCount_);
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::fillArray(values_, AggregationFunctions::avg::template getInitValue<V>(), maxHashCount_);
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="sourceIndices">GPU buffer sourceIndices of existing keys (will be copied to a new buffer)</param>
    /// <param name="stringLengths">GPU buffer stringLengths of existing keys (will be copied to a new buffer)</param>
    /// <param name="keysBuffer">existing GPUString keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, int32_t* stringLengths, GPUMemory::GPUString keysBuffer) : maxHashCount_(maxHashCount)
    {
        int64_t totalCharCount = 0;
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&stringLengths_, maxHashCount_);
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
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            GPUMemory::free(keysBuffer_);
            throw;
        }
        GPUMemory::copyDeviceToDevice(sourceIndices_, sourceIndices, maxHashCount_);
        GPUMemory::copyDeviceToDevice(stringLengths_, stringLengths, maxHashCount_);
        GPUMemory::copyDeviceToDevice(keysBuffer_.stringIndices, keysBuffer.stringIndices, maxHashCount_);
        if (totalCharCount > 0)
        {
            GPUMemory::copyDeviceToDevice(keysBuffer_.allChars, keysBuffer.allChars, totalCharCount);
        }
        GPUMemory::fillArray(values_, AggregationFunctions::avg::template getInitValue<V>(), maxHashCount_);
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        GPUMemory::free(stringLengths_);
        GPUMemory::free(keysBuffer_);

        GPUMemory::free(values_);
        GPUMemory::free(keyOccurenceCount_);
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffer with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void groupBy(GPUMemory::GPUString inKeys, V* inValues, int32_t dataElementCount)
    {
        if (dataElementCount > 0)
        {
            Context& context = Context::getInstance();
            kernel_group_by_string<AggregationFunctions::avg><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                sourceIndices_, stringLengths_, keysBuffer_, values_, keyOccurenceCount_,
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
    int32_t getMaxHashCount()
    {
        return maxHashCount_;
    }


    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurence counts separately
    /// <param name="keysStringLengths">output buffer to fill with reconstructed string lengths of keys</param>
    /// <param name="keysAllChars">output buffer to fill with reconstructed chars of keys</param>
    /// <param name="values">output buffer to fill with reconstructed values</param>
    /// <param name="occurences">output buffer to fill with reconstructed occurences</param>
    /// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void reconstructRawNumbers(std::vector<int32_t>& keysStringLengths,
                               std::vector<char>& keysAllChars,
                               V* values,
                               int64_t* occurences,
                               int32_t* elementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        GPUReconstruct::ReconstructStringColRaw(keysStringLengths, keysAllChars, elementCount,
                                                keysBuffer_, occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructCol(values, elementCount, values_, occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructCol(occurences, elementCount, keyOccurenceCount_, occupancyMask.get(), maxHashCount_);
    }


    /// Get the final results of GROUP BY operation - for operations Min, Max and Sum on single GPU
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
    void getResults(GPUMemory::GPUString* outKeys, O** outValues, int32_t* outDataElementCount)
    {
		static_assert(std::is_integral<V>::value || std::is_floating_point<V>::value,
			"GPUGroupBy<avg>.getResults V (values) must be integral or floating point");

        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        cuda_ptr<O> outValuesGPU(maxHashCount_);
		// Divide by counts to get averages for buckets
		try
		{
			GPUArithmetic::colCol<ArithmeticOperations::div>(outValuesGPU.get(), values_, keyOccurenceCount_, maxHashCount_);
		}
		catch (query_engine_error& err)
		{
			if (err.GetQueryEngineError() != QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR)
			{
				throw err; // Rethrow
			}
			// else ignore, because div by zero is OK here
		}
		// Reonstruct result with original occupancyMask
		GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, outValuesGPU.get(), occupancyMask.get(), maxHashCount_);
        GPUReconstruct::ReconstructStringColKeep(outKeys, outDataElementCount, keysBuffer_,
                                                 occupancyMask.get(), maxHashCount_);
    }


    /// Merge results from all devices and store to buffers on default device (multi GPU function)
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final
    ///   keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated
    ///   and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer
    ///   (will be filled with count of reconstructed elements)</param>
    /// <param name="tables">vector of unique pointers
    ///   to IGroupBy objects with hash tables on every device (GPU)</param>
    void getResults(GPUMemory::GPUString* outKeys,
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

            std::vector<int32_t> keysAllHostStringLengths;
            std::vector<char> keysAllHostAllChars;
            std::vector<V> valuesAllHost;
			std::vector<int64_t> occurencesAllHost;
            int32_t sumElementCount = 0;

            // Collect data from all devices (graphic cards) to host
            for (int i = 0; i < tables.size(); i++)
            {
                if (tables[i].get() == nullptr)
                {
                    break;
                }
                // TODO change to cudaMemcpyPeerAsync
                GPUGroupBy<AggregationFunctions::avg, O, std::string,V>* table =
                    reinterpret_cast<GPUGroupBy<AggregationFunctions::avg, O, std::string,V>*>(tables[i].get());
                std::vector<int32_t> keysStringLengths;
                std::vector<char> keysAllChars;
                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->getMaxHashCount());
                std::unique_ptr<int64_t[]> occurences = std::make_unique<int64_t[]>(table->getMaxHashCount());
                int32_t elementCount;
                Context::getInstance().bindDeviceToContext(i);

                // Reconstruct keys and values
                table->reconstructRawNumbers(keysStringLengths, keysAllChars, values.get(), occurences.get(), &elementCount);

                // Append data to host vectors
                keysAllHostStringLengths.insert(keysAllHostStringLengths.end(),
                                                keysStringLengths.begin(), keysStringLengths.end());
                keysAllHostAllChars.insert(keysAllHostAllChars.end(), keysAllChars.begin(),
                                           keysAllChars.end());
                valuesAllHost.insert(valuesAllHost.end(), values.get(), values.get() + elementCount);
                occurencesAllHost.insert(occurencesAllHost.end(), occurences.get(), occurences.get() + elementCount);
				sumElementCount += elementCount;
            }

            Context::getInstance().bindDeviceToContext(oldDeviceId);
            if (sumElementCount > 0)
            {
                cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
                cuda_ptr<V> valuesAllGPU(sumElementCount);
				cuda_ptr<int64_t> occurencesAllGPU(sumElementCount);

                // Copy the condens from host to default device
                GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                            keysAllHostStringLengths.data(), sumElementCount);
                GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);
                GPUMemory::copyHostToDevice(occurencesAllGPU.get(), occurencesAllHost.data(), sumElementCount);

                // Construct new GPUString with all keys (might be duplicated yet)
                GPUMemory::GPUString keysAllGPU;
                GPUMemory::alloc(&(keysAllGPU.stringIndices), sumElementCount);
                GPUReconstruct::PrefixSum(keysAllGPU.stringIndices, keysAllGPUStringLengths.get(), sumElementCount);
                GPUMemory::alloc(&(keysAllGPU.allChars), keysAllHostAllChars.size());
                GPUMemory::copyHostToDevice(keysAllGPU.allChars, keysAllHostAllChars.data(),
                                            keysAllHostAllChars.size());

                // Merge results
				V* valuesMerged = nullptr;
				int64_t* occurencesMerged = nullptr;

                // Calculate sum of values
				// Initialize new empty sumGroupBy table
                GPUMemory::GPUString keysToDiscard;
				GPUGroupBy<AggregationFunctions::sum, V, std::string, V> sumGroupBy(sumElementCount);
				sumGroupBy.groupBy(keysAllGPU, valuesAllGPU.get(), sumElementCount);
				sumGroupBy.getResults(&keysToDiscard, &valuesMerged, outDataElementCount);
                GPUMemory::free(keysToDiscard);

                // Calculate sum of occurences
				// Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
				GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> countGroupBy(sumElementCount, sumGroupBy.sourceIndices_, sumGroupBy.stringLengths_, sumGroupBy.keysBuffer_);
				countGroupBy.groupBy(keysAllGPU, occurencesAllGPU.get(), sumElementCount);
				countGroupBy.getResults(outKeys, &occurencesMerged, outDataElementCount);

                // Divide merged values by merged occurences to get final averages
				GPUMemory::alloc(outValues, *outDataElementCount);
				GPUArithmetic::colCol<ArithmeticOperations::div>(*outValues, valuesMerged, occurencesMerged, *outDataElementCount);
				
                GPUMemory::free(valuesMerged);
				GPUMemory::free(occurencesMerged);
                GPUMemory::free(keysAllGPU);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};



/// GROUP BY String, specialization for COUNT
template <typename V>
class GPUGroupBy<AggregationFunctions::count, int64_t, std::string, V> : public IGroupBy
{
private:
    int32_t* sourceIndices_ = nullptr;
    int32_t* stringLengths_ = nullptr;
    GPUMemory::GPUString keysBuffer_ {nullptr, nullptr};

    /// Value buffer of the hash table
    V* values_ = nullptr;
    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount) : maxHashCount_(maxHashCount)
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::allocAndSet(&stringLengths_, 0, maxHashCount_);
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::fillArray(values_, AggregationFunctions::count::template getInitValue<V>(), maxHashCount_);
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="keys">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, int32_t* stringLengths, GPUMemory::GPUString keysBuffer) : maxHashCount_(maxHashCount)
    {
        int64_t totalCharCount = 0;
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&stringLengths_, maxHashCount_);
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
            // And for values and occurences
            GPUMemory::alloc(&values_, maxHashCount_);
            GPUMemory::allocAndSet(&keyOccurenceCount_, 0, maxHashCount_);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            GPUMemory::free(keysBuffer_);
            throw;
        }
        GPUMemory::copyDeviceToDevice(sourceIndices_, sourceIndices, maxHashCount_);
        GPUMemory::copyDeviceToDevice(stringLengths_, stringLengths, maxHashCount_);
        GPUMemory::copyDeviceToDevice(keysBuffer_.stringIndices, keysBuffer.stringIndices, maxHashCount_);
        if (totalCharCount > 0)
        {
            GPUMemory::copyDeviceToDevice(keysBuffer_.allChars, keysBuffer.allChars, totalCharCount);
        }
        GPUMemory::fillArray(values_, AggregationFunctions::count::template getInitValue<V>(), maxHashCount_);
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        GPUMemory::free(stringLengths_);
        GPUMemory::free(keysBuffer_);

        GPUMemory::free(values_);
        GPUMemory::free(keyOccurenceCount_);
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffer with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void groupBy(GPUMemory::GPUString inKeys, V* inValues, int32_t dataElementCount)
    {
        if (dataElementCount > 0)
        {
            Context& context = Context::getInstance();
            kernel_group_by_string<AggregationFunctions::count><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                sourceIndices_, stringLengths_, keysBuffer_, values_, keyOccurenceCount_,
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
    int32_t getMaxHashCount()
    {
        return maxHashCount_;
    }


    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurence counts separately
    /// <param name="keys">output buffer to fill with reconstructed keys</param>
    /// <param name="values">not used buffer if using operations COUNT - nullptr can be used</param>
    /// <param name="occurences">output buffer to fill with reconstructed occurences</param>
    /// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void reconstructRawNumbers(std::vector<int32_t>& keysStringLengths,
                               std::vector<char>& keysAllChars,
                               V* values,
                               int64_t* occurences,
                               int32_t* elementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        GPUReconstruct::ReconstructStringColRaw(keysStringLengths, keysAllChars, elementCount,
                                                keysBuffer_, occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructCol(occurences, elementCount, keyOccurenceCount_, occupancyMask.get(), maxHashCount_);
    }


    /// Get the final results of GROUP BY operation - for operations Min, Max and Sum on single GPU
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
    void getResults(GPUMemory::GPUString* outKeys, int64_t** outValues, int32_t* outDataElementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_is_bucket_occupied<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);
        GPUReconstruct::ReconstructStringColKeep(outKeys, outDataElementCount, keysBuffer_,
                                                 occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, keyOccurenceCount_,
                                           occupancyMask.get(), maxHashCount_);
    }


    /// Merge results from all devices and store to buffers on default device (multi GPU function)
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final
    /// keys)</param> <param name="outValues">double pointer of output GPU buffer (will be allocated
    /// and filled with final values)</param> <param name="outDataElementCount">output CPU buffer
    /// (will be filled with count of reconstructed elements)</param> <param name="tables">vector of
    /// unique pointers to IGroupBy objects with hash tables on every device (GPU)</param>
    void getResults(GPUMemory::GPUString* outKeys,
                    int64_t** outValues,
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

            std::vector<int32_t> keysAllHostStringLengths;
            std::vector<char> keysAllHostAllChars;
			std::vector<int64_t> occurencesAllHost;
            int32_t sumElementCount = 0;

            // Collect data from all devices (graphic cards) to host
            for (int i = 0; i < tables.size(); i++)
            {
                if (tables[i].get() == nullptr)
                {
                    break;
                }
                // TODO change to cudaMemcpyPeerAsync
                GPUGroupBy<AggregationFunctions::count, int64_t, std::string, V>* table =
                    reinterpret_cast<GPUGroupBy<AggregationFunctions::count, int64_t, std::string,V>*>(tables[i].get());
                std::vector<int32_t> keysStringLengths;
                std::vector<char> keysAllChars;
                std::unique_ptr<int64_t[]> occurences = std::make_unique<int64_t[]>(table->getMaxHashCount());
				int32_t elementCount;
                Context::getInstance().bindDeviceToContext(i);

                // Reconstruct keys and values
                table->reconstructRawNumbers(keysStringLengths, keysAllChars, nullptr, occurences.get(), &elementCount);

                // Append data to host vectors
                keysAllHostStringLengths.insert(keysAllHostStringLengths.end(),
                                                keysStringLengths.begin(), keysStringLengths.end());
                keysAllHostAllChars.insert(keysAllHostAllChars.end(), keysAllChars.begin(),
                                           keysAllChars.end());
                occurencesAllHost.insert(occurencesAllHost.end(), occurences.get(), occurences.get() + elementCount);
				sumElementCount += elementCount;
            }

            Context::getInstance().bindDeviceToContext(oldDeviceId);
            if (sumElementCount > 0)
            {
                cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
				cuda_ptr<int64_t> occurencesAllGPU(sumElementCount);

                // Copy the condens from host to default device
                GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                            keysAllHostStringLengths.data(), sumElementCount);
				GPUMemory::copyHostToDevice(occurencesAllGPU.get(), occurencesAllHost.data(), sumElementCount);

                // Construct new GPUString
                GPUMemory::GPUString keysAllGPU;
                GPUMemory::alloc(&(keysAllGPU.stringIndices), sumElementCount);
                GPUReconstruct::PrefixSum(keysAllGPU.stringIndices, keysAllGPUStringLengths.get(), sumElementCount);
                GPUMemory::alloc(&(keysAllGPU.allChars), keysAllHostAllChars.size());
                GPUMemory::copyHostToDevice(keysAllGPU.allChars, keysAllHostAllChars.data(),
                                            keysAllHostAllChars.size());

                // Merge results
                GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> finalGroupBy(sumElementCount);
                finalGroupBy.groupBy(keysAllGPU, occurencesAllGPU.get(), sumElementCount);
                finalGroupBy.getResults(outKeys, outValues, outDataElementCount);

                GPUMemory::free(keysAllGPU);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};
