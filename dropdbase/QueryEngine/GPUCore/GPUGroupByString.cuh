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
                                       int8_t* valuesNullMaskUncompressed,
                                       int64_t* keyOccurrenceCount,
                                       int32_t loweredMaxHashCount,
                                       GPUMemory::GPUString inKeys,
                                       V* inValues,
                                       int32_t dataElementCount,
                                       int32_t* errorFlag,
                                       int8_t* inKeysNullMask,
                                       int8_t* inValuesNullMask)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        const int32_t bitMaskIdx = (i / (sizeof(int8_t) * 8));
        const int32_t shiftIdx = (i % (sizeof(int8_t) * 8));
        const bool nullKey = inKeysNullMask ? ((inKeysNullMask[bitMaskIdx] >> shiftIdx) & 1) : false;
        const bool nullValue = inValuesNullMask ? ((inValuesNullMask[bitMaskIdx] >> shiftIdx) & 1) : false;
        int32_t foundIndex = -1;
        int32_t inKeyLength = 0;

        if (!nullKey)
        {
            const int64_t inKeyIndex = GetStringIndex(inKeys.stringIndices, i);
            inKeyLength = GetStringLength(inKeys.stringIndices, i);
            char* inKeyChars = inKeys.allChars + inKeyIndex;
            // Calculate hash
            const int32_t hash = abs(GetHash(inKeyChars, inKeyLength)) % loweredMaxHashCount;

            for (int32_t j = 0; j < loweredMaxHashCount; j++)
            {
                // Calculate index to hash-table from hash
                const int32_t index = ((hash + j) % loweredMaxHashCount) + 1;

                // Check if key is not empty and key is not equal to the currently inserted key
                if (sourceIndices[index] != GBS_SOURCE_INDEX_EMPTY_KEY &&
                    IsNewKey(inKeyChars, inKeyLength, inKeys, keysBuffer, sourceIndices, index))
                {
                    continue; // Hash collision - try to find another index
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
        }
        else
        {
            foundIndex = 0;
            sourceIndices[foundIndex] = GBS_SOURCE_INDEX_KEY_IN_BUFFER;
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
            if (!nullValue)
            {
                // Aggregate value
                if (values)
                {
                    AGG{}(&values[foundIndex], inValues[i]);
                    // if (inValuesNullMask) - probably not necessary, if sure, delete
                    valuesNullMaskUncompressed[foundIndex] = 0;
                }
                // Increment occurrence counter
                if (keyOccurrenceCount)
                {
                    atomicAdd(reinterpret_cast<cuUInt64*>(&keyOccurrenceCount[foundIndex]), 1);
                }
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
    static constexpr bool USE_VALUES = std::is_same<AGG, AggregationFunctions::min>::value ||
                                       std::is_same<AGG, AggregationFunctions::max>::value ||
                                       std::is_same<AGG, AggregationFunctions::sum>::value ||
                                       std::is_same<AGG, AggregationFunctions::avg>::value;

    static constexpr bool USE_KEY_OCCURRENCES = std::is_same<AGG, AggregationFunctions::avg>::value ||
                                                std::is_same<AGG, AggregationFunctions::count>::value;

    static constexpr bool DIRECT_VALUES = std::is_same<AGG, AggregationFunctions::min>::value ||
                                          std::is_same<AGG, AggregationFunctions::max>::value ||
                                          std::is_same<AGG, AggregationFunctions::sum>::value;

public:
    /// Temp buffer where one value points to input key
    /// or tells the key is already in keysBuffer_
    int32_t* sourceIndices_ = nullptr;
    /// Buffer with lengths of collected string keys
    int32_t* stringLengths_ = nullptr;
    /// Buffer with collected string keys
    GPUMemory::GPUString keysBuffer_{nullptr, nullptr};

private:
    /// Value buffer of the hash table
    V* values_ = nullptr;
    int8_t* valuesNullMaskUncompressed_ = nullptr;

    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurrenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    const int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurrence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount) : maxHashCount_(maxHashCount + 1) // + 1 for NULL key
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
                GPUMemory::allocAndSet(&valuesNullMaskUncompressed_, 1, maxHashCount_);
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
            if (valuesNullMaskUncompressed_)
            {
                GPUMemory::free(valuesNullMaskUncompressed_);
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
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, int32_t* stringLengths, GPUMemory::GPUString keysBuffer)
    : GPUGroupBy(maxHashCount)
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
            GPUMemory::free(valuesNullMaskUncompressed_);
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
    void ProcessBlock(GPUMemory::GPUString inKeys,
                      V* inValues,
                      int32_t dataElementCount,
                      int8_t* inKeysNullMask = nullptr,
                      int8_t* inValuesNullMask = nullptr)
    {
        if (dataElementCount > 0)
        {
            Context& context = Context::getInstance();
            kernel_group_by_string<AGG><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                sourceIndices_, stringLengths_, keysBuffer_, values_, valuesNullMaskUncompressed_,
                keyOccurrenceCount_, maxHashCount_ - 1, inKeys, inValues, dataElementCount,
                errorFlagSwapper_.GetFlagPointer(), inKeysNullMask, inValuesNullMask);
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

    /// Create memory-wasting null mask for keys - one 1 at [0], other zeros
    cuda_ptr<int8_t> CreateKeyNullMask()
    {
        cuda_ptr<int8_t> keyNullMask(maxHashCount_, 0);
        GPUMemory::memset(keyNullMask.get(), 1, 1);
        return keyNullMask;
    }

    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurrence counts separately
    /// <param name="keys">output buffer to fill with reconstructed keys</param>
    /// <param name="values">output buffer to fill with reconstructed values</param>
    /// <param name="occurrences">not used buffer if using operations MIN, MAX or SUM - nullptr can be used</param>
    /// <param name="elementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void ReconstructRawNumbers(std::vector<int32_t>& keysStringLengths,
                               std::vector<char>& keysAllChars,
                               int8_t* keysNullMask,
                               V* values,
                               int8_t* valuesNullMask,
                               int64_t* occurrences,
                               int32_t* elementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        cuda_ptr<int8_t> keysNullMaskInput = CreateKeyNullMask();

        GPUReconstruct::ReconstructStringColRaw(keysStringLengths, keysAllChars, elementCount,
                                                keysBuffer_, occupancyMask.get(), maxHashCount_);
        GPUReconstruct::reconstructCol(keysNullMask, elementCount, keysNullMaskInput.get(),
                                       occupancyMask.get(), maxHashCount_);

        if (USE_VALUES)
        {
            GPUReconstruct::reconstructCol(values, elementCount, values_, occupancyMask.get(), maxHashCount_);
            GPUReconstruct::reconstructCol(valuesNullMask, elementCount, valuesNullMaskUncompressed_,
                                           occupancyMask.get(), maxHashCount_);
        }
        if (USE_KEY_OCCURRENCES)
        {
            GPUReconstruct::reconstructCol(occurrences, elementCount, keyOccurrenceCount_,
                                           occupancyMask.get(), maxHashCount_);
        }
    }


    /// Get the final results of GROUP BY operation - for operations Min, Max and Sum on single GPU
    /// <param name="outKeys">pointer to GPUString struct (will be allocated and filled with final keys)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count of reconstructed elements)</param>
    void GetResults(GPUMemory::GPUString* outKeys,
                    O** outValues,
                    int32_t* outDataElementCount,
                    int8_t** outKeysNullMask = nullptr,
                    int8_t** outValuesNullMask = nullptr)
    {
        Context& context = Context::getInstance();

        // Create buffer for bucket compression - reconstruction
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        // Compute occupancyMask
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        cuda_ptr<int8_t> keysNullMaskCompressed =
            GPUReconstruct::CompressNullMask(CreateKeyNullMask().get(), maxHashCount_);

        GPUReconstruct::ReconstructStringColKeep(outKeys, outDataElementCount, keysBuffer_,
                                                 occupancyMask.get(), maxHashCount_, outKeysNullMask,
                                                 outKeysNullMask ? keysNullMaskCompressed.get() : nullptr);

        if (USE_VALUES)
        {
            cuda_ptr<int8_t> valuesNullMaskCompressed((maxHashCount_ + sizeof(int32_t) * 8 - 1) /
                                                          (sizeof(int8_t) * 8),
                                                      0);
            int8_t* valuesNullMaskCompressedPtr =
                (outValuesNullMask ? valuesNullMaskCompressed.get() : nullptr);
            if (valuesNullMaskCompressedPtr)
            {
                kernel_compress_null_mask<<<Context::getInstance().calcGridDim(maxHashCount_),
                                            Context::getInstance().getBlockDim()>>>(
                    reinterpret_cast<int32_t*>(valuesNullMaskCompressedPtr),
                    valuesNullMaskUncompressed_, maxHashCount_);
            }

            // Reconstruct aggregated values
            if (DIRECT_VALUES) // for min, max and sum: values_ are direct results, just reconstruct them
            {
                if (!std::is_same<O, V>::value)
                {
                    CheckQueryEngineError(GPU_EXTENSION_ERROR, "Input and output value data type "
                                                               "must be the same in GROUP BY");
                }
                // reinterpret_cast is needed to solve compilation error
                GPUReconstruct::reconstructColKeep(outValues, outDataElementCount,
                                                   reinterpret_cast<O*>(values_), occupancyMask.get(), maxHashCount_,
                                                   outValuesNullMask, valuesNullMaskCompressedPtr);
            }
            else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg: values_ need to be divided by keyOccurrences_ and reconstructed
            {
                cuda_ptr<O> outValuesGPU(maxHashCount_);
                // Divide by counts to get averages for buckets
                try
                {
                    GPUArithmetic::colCol<ArithmeticOperations::div>(outValuesGPU.get(), values_,
                                                                     keyOccurrenceCount_, maxHashCount_);
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
                GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, outValuesGPU.get(),
                                                   occupancyMask.get(), maxHashCount_,
                                                   outValuesNullMask, valuesNullMaskCompressedPtr);
            }
        }
        else // for count: reconstruct and return keyOccurrences_
        {
            if (!std::is_same<O, int64_t>::value)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR, "Output value data type must be int64_t "
                                                           "for COUNT-GROUP BY operation");
            }
            // reinterpret_cast is needed to solve compilation error
            // not reinterpreting anything here actually, outValues is int64_t** always in this else-branch
            GPUReconstruct::reconstructColKeep(reinterpret_cast<int64_t**>(outValues), outDataElementCount,
                                               keyOccurrenceCount_, occupancyMask.get(), maxHashCount_);
            if (outValuesNullMask)
            {
                GPUMemory::allocAndSet(outValuesNullMask, 0,
                                       (*outDataElementCount + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8));
            }
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
                    std::vector<std::unique_ptr<IGroupBy>>& tables,
                    int8_t** outKeysNullMask = nullptr,
                    int8_t** outValuesNullMask = nullptr)
    {
        if (tables.size() <= 0) // invalid count of tables
        {
            throw std::invalid_argument("Number of tables have to be at least 1.");
        }
        else if (tables.size() == 1 || tables[1].get() == nullptr) // just one table
        {
            GetResults(outKeys, outValues, outDataElementCount, outKeysNullMask, outValuesNullMask);
        }
        else // more tables
        {
            int oldDeviceId = Context::getInstance().getBoundDeviceID();

            std::vector<int32_t> keysAllHostStringLengths;
            std::vector<char> keysAllHostAllChars;
            std::vector<int8_t> keysNullMaskAllHost;

            std::vector<V> valuesAllHost;
            std::vector<int8_t> valuesNullMaskAllHost;

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
                GPUGroupBy<AGG, O, std::string, V>* table =
                    reinterpret_cast<GPUGroupBy<AGG, O, std::string, V>*>(tables[i].get());

                std::vector<int32_t> keysStringLengths;
                std::vector<char> keysAllChars;
                std::unique_ptr<int8_t[]> keysNullMask =
                    std::make_unique<int8_t[]>(table->GetMaxHashCount());

                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->GetMaxHashCount());
                std::unique_ptr<int8_t[]> valuesNullMask =
                    std::make_unique<int8_t[]>(table->GetMaxHashCount());

                std::unique_ptr<int64_t[]> occurrences =
                    std::make_unique<int64_t[]>(table->GetMaxHashCount());

                int32_t elementCount;
                Context::getInstance().bindDeviceToContext(i);

                // Reconstruct keys and values
                table->ReconstructRawNumbers(keysStringLengths, keysAllChars, keysNullMask.get(),
                                             values.get(), valuesNullMask.get(), occurrences.get(),
                                             &elementCount);

                // Append data to host vectors
                keysAllHostStringLengths.insert(keysAllHostStringLengths.end(),
                                                keysStringLengths.begin(), keysStringLengths.end());
                keysAllHostAllChars.insert(keysAllHostAllChars.end(), keysAllChars.begin(),
                                           keysAllChars.end());

                keysNullMaskAllHost.insert(keysNullMaskAllHost.end(), keysNullMask.get(),
                                           keysNullMask.get() + elementCount);

                if (USE_VALUES)
                {
                    valuesAllHost.insert(valuesAllHost.end(), values.get(), values.get() + elementCount);
                    valuesNullMaskAllHost.insert(valuesNullMaskAllHost.end(), valuesNullMask.get(),
                                                 valuesNullMask.get() + elementCount);
                }
                if (USE_KEY_OCCURRENCES)
                {
                    occurrencesAllHost.insert(occurrencesAllHost.end(), occurrences.get(),
                                              occurrences.get() + elementCount);
                }
                sumElementCount += elementCount;
            }

            Context::getInstance().bindDeviceToContext(oldDeviceId);
            if (sumElementCount > 0)
            {
                cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
                cuda_ptr<int8_t> keysNullMaskAllGPU(sumElementCount);

                cuda_ptr<V> valuesAllGPU(sumElementCount);
                cuda_ptr<int8_t> valuesNullMaskAllGPU(sumElementCount);

                cuda_ptr<int64_t> occurrencesAllGPU(sumElementCount);

                // Copy the condens from host to default device
                GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                            keysAllHostStringLengths.data(), sumElementCount);
                GPUMemory::copyHostToDevice(keysNullMaskAllGPU.get(), keysNullMaskAllHost.data(), sumElementCount);

                if (USE_VALUES)
                {
                    GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);
                    GPUMemory::copyHostToDevice(valuesNullMaskAllGPU.get(),
                                                valuesNullMaskAllHost.data(), sumElementCount);
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
                    GPUGroupBy<AGG, O, std::string, V> finalGroupBy(sumElementCount);
                    finalGroupBy.ProcessBlock(
                        keysAllGPU, valuesAllGPU.get(), sumElementCount,
                        GPUReconstruct::CompressNullMask(keysNullMaskAllGPU.get(), sumElementCount).get(),
                        GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                            .get());
                    finalGroupBy.GetResults(outKeys, outValues, outDataElementCount,
                                            outKeysNullMask, outValuesNullMask);
                }
                else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg
                {
                    V* valuesMerged = nullptr;
                    int64_t* occurrencesMerged = nullptr;

                    // Calculate sum of values
                    // Initialize new empty sumGroupBy table
                    GPUMemory::GPUString keysToDiscard;
                    GPUGroupBy<AggregationFunctions::sum, V, std::string, V> sumGroupBy(sumElementCount);
                    sumGroupBy.ProcessBlock(
                        keysAllGPU, valuesAllGPU.get(), sumElementCount,
                        GPUReconstruct::CompressNullMask(keysNullMaskAllGPU.get(), sumElementCount).get(),
                        GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                            .get());
                    sumGroupBy.GetResults(&keysToDiscard, &valuesMerged, outDataElementCount,
                                          outKeysNullMask, outValuesNullMask);
                    GPUMemory::free(keysToDiscard);
                    // Don't need these results, will be computed again in countGroupBy - TODO multi-value GroupBy
                    if (outKeysNullMask) // if used (if double pointer is not nullptr)
                    {
                        GPUMemory::free(*outKeysNullMask); // free array
                    }
                    if (outValuesNullMask)
                    {
                        GPUMemory::free(*outValuesNullMask);
                    }

                    // Calculate sum of occurrences
                    // Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> countGroupBy(
                        sumElementCount, sumGroupBy.sourceIndices_, sumGroupBy.stringLengths_,
                        sumGroupBy.keysBuffer_);
                    countGroupBy.ProcessBlock(
                        keysAllGPU, occurrencesAllGPU.get(), sumElementCount,
                        GPUReconstruct::CompressNullMask(keysNullMaskAllGPU.get(), sumElementCount).get(),
                        GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                            .get());
                    countGroupBy.GetResults(outKeys, &occurrencesMerged, outDataElementCount,
                                            outKeysNullMask, outValuesNullMask);

                    // Divide merged values by merged occurrences to get final averages
                    GPUMemory::alloc(outValues, *outDataElementCount);
                    try
                    {
                        GPUArithmetic::colCol<ArithmeticOperations::div>(*outValues, valuesMerged, occurrencesMerged,
                                                                         *outDataElementCount);
                    }
                    catch (const query_engine_error& err)
                    {
                        // Rethrow just if error is not division by zero.
                        // Division by zero is OK here because some values could be NULL
                        // and therefore keyOccurrences could be 0.
                        if (err.GetQueryEngineError() != QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR)
                        {
                            throw err;
                        }
                    }

                    GPUMemory::free(valuesMerged);
                    GPUMemory::free(occurrencesMerged);
                }
                else // for count
                {
                    if (!std::is_same<O, int64_t>::value)
                    {
                        CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                              "Output value data type in GROUP BY with COUNT must "
                                              "be int64_t");
                    }
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::string, int64_t> finalGroupBy(sumElementCount);
                    finalGroupBy.ProcessBlock(
                        keysAllGPU, occurrencesAllGPU.get(), sumElementCount,
                        GPUReconstruct::CompressNullMask(keysNullMaskAllGPU.get(), sumElementCount).get(),
                        nullptr);
                    // reinterpret_cast is needed to solve compilation error
                    finalGroupBy.GetResults(outKeys, reinterpret_cast<int64_t**>(outValues),
                                            outDataElementCount, outKeysNullMask, outValuesNullMask);
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
