#pragma once

#include <stdio.h>

#include "../../DataType.h"
#include "../GPUError.h"
#include "GPUGroupByString.cuh"

struct CPUString
{
    std::vector<int32_t> stringLengths;
    std::vector<char> allChars;
};

/// Compute hash of multi-key
__device__ int32_t GetHash(DataType* keyTypes, int32_t keysColCount, void** inKeys, int32_t i, int32_t detlaHash);

/// Chceck for equality of two multi-keys
__device__ bool
AreEqualMultiKeys(DataType* keyTypes, int32_t keysColCount, void** keysA, int32_t indexA, void** keysB, int32_t indexB);

/// Check if multi-key is new (not present in keys buffer nor in sourceIndices)
/// Input multi-key is defined by inKeys and i
/// <param name="sourceIndices">points to inKeys</param>
/// <param name="index">points to keysBuffer</param>
__device__ bool IsNewMultiKey(DataType* keyTypes,
                              int32_t keysColCount,
                              void** inKeys,
                              int32_t i,
                              void** keysBuffer,
                              int32_t* sourceIndices,
                              int32_t index);

/// Reconstruct one key column (keep on GPU)
template <typename T>
void ReconstructSingleKeyColKeep(std::vector<void*>* outKeysVector,
                                 int32_t* outDataElementCount,
                                 int8_t* occupancyMaskPtr,
                                 void** keyCol,
                                 int32_t elementCount)
{
    // Copy key col pointer to CPU
    T* keyBufferSingleCol;
    GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<T**>(keyCol), 1);

    // Reconstruct key col
    T* outKeysSingleCol;
    GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount, keyBufferSingleCol,
                                       occupancyMaskPtr, elementCount);

    outKeysVector->emplace_back(outKeysSingleCol);
}

/// Reconstruct one string key column (keep on GPU)
template <>
void ReconstructSingleKeyColKeep<std::string>(std::vector<void*>* outKeysVector,
                                              int32_t* outDataElementCount,
                                              int8_t* occupancyMaskPtr,
                                              void** keyCol,
                                              int32_t elementCount);

/// Reconstruct one key column to CPU
template <typename T>
void ReconstructSingleKeyCol(std::vector<void*>* outKeysVector,
                             int32_t* outDataElementCount,
                             int8_t* occupancyMaskPtr,
                             void** keyCol,
                             int32_t elementCount)
{
    // Copy key col pointer to CPU
    T* keyBufferSingleCol;
    GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<T**>(keyCol), 1);

    // Get out count to allocate CPU memory
    cuda_ptr<int32_t> prefixSumPtr(elementCount);
    GPUReconstruct::PrefixSum(prefixSumPtr.get(), occupancyMaskPtr, elementCount);
    int32_t outCount;
    GPUMemory::copyDeviceToHost(&outCount, prefixSumPtr.get() + elementCount - 1, 1);

    // Allocate CPU memory, reconstruct key col and copy
    T* outKeysSingleCol = new T[outCount];
    GPUReconstruct::reconstructCol(outKeysSingleCol, outDataElementCount, keyBufferSingleCol,
                                   occupancyMaskPtr, elementCount);

    outKeysVector->emplace_back(outKeysSingleCol);
}

/// Reconstruct one string key column to CPU
template <>
void ReconstructSingleKeyCol<std::string>(std::vector<void*>* outKeysVector,
                                          int32_t* outDataElementCount,
                                          int8_t* occupancyMaskPtr,
                                          void** keyCol,
                                          int32_t elementCount);

/// Alloc 2-dimensional buffer for multi-keys storage
void AllocKeysBuffer(void*** keysBuffer,
                     std::vector<DataType> keyTypes,
                     int32_t rowCount,
                     std::vector<void*>* pointers = nullptr);

/// Free 2-dimensional buffer for multi-keys storage
void FreeKeysBuffer(void** keysBuffer, DataType* keyTypes, int32_t keysColCount);

/// Free buffers from vector
void FreeKeysVector(std::vector<void*> keysVector, std::vector<DataType> keyTypes);

/// GROUP BY Kernel processes input (inKeys and inValues). New keys from inKeys are added
/// to the hash table and values from inValues are aggregated.
template <typename AGG, typename V>
__global__ void kernel_group_by_multi_key(DataType* keyTypes,
                                          const int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          V* values,
                                          int64_t* keyOccurrenceCount,
                                          const int32_t maxHashCount,
                                          void** inKeys,
                                          V* inValues,
                                          const int32_t dataElementCount,
                                          const int32_t arrayMultiplier,
                                          const int32_t hashCoef,
                                          int32_t* errorFlag)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Calculate hash
        const int32_t hash = abs(GetHash(keyTypes, keysColCount, inKeys, i, hashCoef)) % maxHashCount;

        int32_t foundIndex = -1;
        for (int32_t j = 0; j < maxHashCount; j++)
        {
            // Calculate index to hash-table from hash
            const int32_t index = (hash + j) % maxHashCount;

            // Check if key is not empty and key is not equal to the currently inserted key
            if (sourceIndices[index] != GBS_SOURCE_INDEX_EMPTY_KEY &&
                IsNewMultiKey(keyTypes, keysColCount, inKeys, i, keysBuffer, sourceIndices, index))
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
                    IsNewMultiKey(keyTypes, keysColCount, inKeys, i, keysBuffer, sourceIndices, index))
                {
                    continue; // Try to find another index
                }
            }
            else if (sourceIndices[index] != i &&
                     IsNewMultiKey(keyTypes, keysColCount, inKeys, i, keysBuffer, sourceIndices, index))
            {
                continue; // try to find another index
            }

            // The key was added or found as already existing
            foundIndex = index;
            break;
        }

        // If no index was found - the hash table is full
        if (foundIndex == -1)
        {
            atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_HASH_TABLE_FULL));
        }
        else // else - if we found a valid index
        {
            // Use aggregation of values on the bucket and the corresponding counter
            if (values)
            {
                AGG{}(values + foundIndex * arrayMultiplier + threadIdx.x % arrayMultiplier, inValues[i]);
            }
            if (keyOccurrenceCount)
            {
                atomicAdd(reinterpret_cast<cuUInt64*>(keyOccurrenceCount + foundIndex * arrayMultiplier +
                                                      threadIdx.x % arrayMultiplier),
                          1);
            }
        }
    }
}

/// Collect string lengths according to sourceIndices
__global__ void kernel_collect_string_lengths(int32_t* stringLengths,
                                              int32_t* sourceIndices,
                                              GPUMemory::GPUString** inKeysSingleCol,
                                              GPUMemory::GPUString** keysBufferSingleCol,
                                              int32_t maxHashCount);

/// Collect multi keys to keysBuffer according to sourceIndices
__global__ void kernel_collect_multi_keys(DataType* keyTypes,
                                          int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          GPUMemory::GPUString* stringSideBuffers,
                                          int32_t** stringLengthsBuffers,
                                          int32_t maxHashCount,
                                          void** inKeys);


/// GROUP BY class for multi-keys
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::vector<void*>, V> : public IGroupBy
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
    /// Indices to input keys - because of atomicity
    int32_t* sourceIndices_ = nullptr;
    /// Keys buffer - all found combination of keys are stored here
    void** keysBuffer_ = nullptr;

private:
    /// Types of keys
    DataType* keyTypes_ = nullptr;
    /// Count of key columns
    const int32_t keysColCount_;
    /// Indices of string key columns
    std::vector<int32_t> stringKeyColIds_;

    /// Value buffer of the hash table
    V* values_ = nullptr;
    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurrenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    const int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurrence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="keyTypes">key column types (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, std::vector<DataType> keyTypes)
    : maxHashCount_(maxHashCount), keysColCount_(keyTypes.size())
    {
        const size_t multipliedCount = static_cast<size_t>(maxHashCount_) * GB_ARRAY_MULTIPLIER;
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&keyTypes_, keysColCount_);
            AllocKeysBuffer(&keysBuffer_, keyTypes, maxHashCount_);
            for (int32_t i = 0; i < keyTypes.size(); i++)
            {
                if (keyTypes[i] == COLUMN_STRING)
                {
                    stringKeyColIds_.emplace_back(i);
                }
            }

            // And for values and occurrences
            if (USE_VALUES)
            {
                GPUMemory::alloc(&values_, multipliedCount);
            }
            if (USE_KEY_OCCURRENCES)
            {
                GPUMemory::allocAndSet(&keyOccurrenceCount_, 0, multipliedCount);
            }
        }
        catch (...)
        {
            if (sourceIndices_)
            {
                GPUMemory::free(sourceIndices_);
            }
            if (keyTypes_)
            {
                GPUMemory::free(keyTypes_);
            }
            if (keysBuffer_)
            {
                for (int32_t i = 0; i < keysColCount_; i++)
                {
                    void* ptr;
                    GPUMemory::copyDeviceToHost(&ptr, keysBuffer_ + i, 1);
                    if (ptr)
                    {
                        GPUMemory::free(ptr);
                    }
                }
                GPUMemory::free(keysBuffer_);
            }
            if (values_)
            {
                GPUMemory::free(values_);
            }
            if (keyOccurrenceCount_)
            {
                GPUMemory::free(keyOccurrenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::copyHostToDevice(keyTypes_, keyTypes.data(), keysColCount_);
        if (USE_VALUES)
        {
            GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), multipliedCount);
        }
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="keyTypes">key column types (will be copied to a new buffer)</param>
    /// <param name="sourceIndices">GPU buffer with existing sourceIndices (will be copied to a new buffer)</param>
    /// <param name="keysBuffer">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, std::vector<DataType> keyTypes, int32_t* sourceIndices, void** keysBuffer)
    : GPUGroupBy(maxHashCount, keyTypes)
    {
        // Copy source indices
        GPUMemory::copyDeviceToDevice(sourceIndices_, sourceIndices, maxHashCount_);

        // Copy all keys (deep copy)
        for (int32_t i = 0; i < keyTypes.size(); i++)
        {
            void* myCol;
            GPUMemory::copyDeviceToHost(&myCol, keysBuffer_ + i, 1);
            void* srcCol;
            GPUMemory::copyDeviceToHost(&srcCol, keysBuffer + i, 1);

            switch (keyTypes[i])
            {
            case DataType::COLUMN_INT:
            {
                GPUMemory::copyDeviceToDevice(reinterpret_cast<int32_t*>(myCol),
                                              reinterpret_cast<int32_t*>(srcCol), maxHashCount_);
                break;
            }
            case DataType::COLUMN_LONG:
            {
                GPUMemory::copyDeviceToDevice(reinterpret_cast<int64_t*>(myCol),
                                              reinterpret_cast<int64_t*>(srcCol), maxHashCount_);
                break;
            }
            case DataType::COLUMN_FLOAT:
            {
                GPUMemory::copyDeviceToDevice(reinterpret_cast<float*>(myCol),
                                              reinterpret_cast<float*>(srcCol), maxHashCount_);
                break;
            }
            case DataType::COLUMN_DOUBLE:
            {
                GPUMemory::copyDeviceToDevice(reinterpret_cast<double*>(myCol),
                                              reinterpret_cast<double*>(srcCol), maxHashCount_);
                break;
            }
            case DataType::COLUMN_STRING:
            {
                // Get source struct
                GPUMemory::GPUString srcString;
                GPUMemory::copyDeviceToHost(&srcString, reinterpret_cast<GPUMemory::GPUString*>(srcCol), 1);

                // Get total char count
                int64_t totalCharCount;
                GPUMemory::copyDeviceToHost(&totalCharCount, srcString.stringIndices + maxHashCount_ - 1, 1);

                // Create my struct
                GPUMemory::GPUString myString;
                GPUMemory::alloc(&(myString.stringIndices), maxHashCount_);
                GPUMemory::alloc(&(myString.allChars), totalCharCount);

                // Copy indices and chars
                GPUMemory::copyDeviceToDevice(myString.stringIndices, srcString.stringIndices, maxHashCount_);
                GPUMemory::copyDeviceToDevice(myString.allChars, srcString.allChars, totalCharCount);

                // Copy struct
                GPUMemory::copyHostToDevice<GPUMemory::GPUString>(reinterpret_cast<GPUMemory::GPUString*>(myCol),
                                                                  &myString, 1);
                break;
            }
            case DataType::COLUMN_INT8_T:
            {
                GPUMemory::copyDeviceToDevice(reinterpret_cast<int8_t*>(myCol),
                                              reinterpret_cast<int8_t*>(srcCol), maxHashCount_);
                break;
            }
            default:
                break;
            }
        }
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        FreeKeysBuffer(keysBuffer_, keyTypes_, keysColCount_);
        GPUMemory::free(keyTypes_);
        if (USE_VALUES)
        {
            GPUMemory::free(values_);
        }
        if (USE_KEY_OCCURRENCES)
        {
            GPUMemory::free(keyOccurrenceCount_);
        }
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeysVector">input vector of buffers with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void GroupBy(std::vector<void*> inKeysVector, V* inValues, int32_t dataElementCount)
    {
        if (dataElementCount > 0)
        {
            if (inKeysVector.size() != keysColCount_)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                      "Incorrect number of key columns in GroupBy");
            }
            Context& context = Context::getInstance();
            // Convert vector to GPU void
            cuda_ptr<void*> inKeys(keysColCount_);
            GPUMemory::copyHostToDevice(inKeys.get(), inKeysVector.data(), keysColCount_);

            // Run group by kernel (get sourceIndices and aggregate values).
            // Parameter hashCoef is comptued as n-th root of maxHashCount, where n is a number of key columns
            kernel_group_by_multi_key<AGG><<<context.calcGridDim(dataElementCount), 768>>>(
                keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, values_, keyOccurrenceCount_,
                maxHashCount_, inKeys.get(), inValues, dataElementCount, GB_ARRAY_MULTIPLIER,
                static_cast<int32_t>(powf(maxHashCount_, 1.0f / keysColCount_)),
                errorFlagSwapper_.GetFlagPointer());
            errorFlagSwapper_.Swap();

            cuda_ptr<int32_t*> stringLengthsBuffers(keysColCount_, 0); // alloc pointers and set to nullptr
            cuda_ptr<GPUMemory::GPUString> stringSideBuffers(keysColCount_, 0); // alloc clean structs on gpu

            for (int32_t t : stringKeyColIds_)
            {
                int32_t* stringLengths;
                GPUMemory::alloc(&stringLengths, maxHashCount_);
                GPUMemory::copyHostToDevice(stringLengthsBuffers.get() + t, &stringLengths, 1); // copy pointer to stringLengths
                kernel_collect_string_lengths<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                    stringLengths, sourceIndices_,
                    reinterpret_cast<GPUMemory::GPUString**>(inKeys.get() + t),
                    reinterpret_cast<GPUMemory::GPUString**>(keysBuffer_ + t), maxHashCount_);

                GPUMemory::GPUString cpuStruct;
                GPUMemory::alloc(&(cpuStruct.stringIndices), maxHashCount_);
                GPUReconstruct::PrefixSum(cpuStruct.stringIndices, stringLengths, maxHashCount_);
                int64_t totalCharCount;
                GPUMemory::copyDeviceToHost(&totalCharCount, cpuStruct.stringIndices + maxHashCount_ - 1, 1);
                GPUMemory::alloc(&(cpuStruct.allChars), totalCharCount);
                GPUMemory::copyHostToDevice(reinterpret_cast<GPUMemory::GPUString*>(
                                                stringSideBuffers.get() + t),
                                            &cpuStruct, 1);
            }

            // Collect multi-keys from inKeys according to sourceIndices
            kernel_collect_multi_keys<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, stringSideBuffers.get(),
                stringLengthsBuffers.get(), maxHashCount_, inKeys.get());
            for (int32_t t : stringKeyColIds_)
            {
                // Free string lengths
                int32_t* stringLengths;
                GPUMemory::copyDeviceToHost(&stringLengths, stringLengthsBuffers.get() + t, 1);
                GPUMemory::free(stringLengths);

                // Free old key buffer for single string col
                GPUMemory::GPUString* structPointer;
                GPUMemory::copyDeviceToHost(&structPointer,
                                            reinterpret_cast<GPUMemory::GPUString**>(keysBuffer_ + t), 1);
                GPUMemory::GPUString cpuStruct;
                GPUMemory::copyDeviceToHost(&cpuStruct, structPointer, 1);
                GPUMemory::free(cpuStruct);

                // And replace key buffer with side buffer
                GPUMemory::copyDeviceToDevice(
                    structPointer, reinterpret_cast<GPUMemory::GPUString*>(stringSideBuffers.get() + t), 1);
            }

            CheckCudaError(cudaGetLastError());
        }
    }


    /// Get the size of the hash table (max. count of unique multi-keys)
    /// <returns>size of the hash table</returns>
    int32_t GetMaxHashCount()
    {
        return maxHashCount_;
    }

    /// Reconstruct needed raw fields (do not calculate final results yet)
    /// Reconstruct keys, values and key occurence counts separately
    /// <param name="multiKeys">output vector of buffers to fill with reconstructed keys</param>
    /// <param name="values">output buffer to fill with reconstructed values</param>
    /// <param name="occurrences">output buffer to fill with reconstructed occurrences</param>
    /// <param name="outDataElementCount">ouptut buffer to fill with element count (one int32_t number)</param>
    void ReconstructRawNumbers(std::vector<void*>& multiKeys, V* values, int64_t* occurrences, int32_t* outDataElementCount)
    {
        Context& context = Context::getInstance();
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        // Copy data types back from GPU
        std::unique_ptr<DataType[]> keyTypesHost = std::make_unique<DataType[]>(keysColCount_);
        GPUMemory::copyDeviceToHost(keyTypesHost.get(), keyTypes_, keysColCount_);

        // Reconstruct keys from all collected cols
        for (int32_t t = 0; t < keysColCount_; t++)
        {
            switch (keyTypesHost[t])
            {
            case DataType::COLUMN_INT:
            {
                ReconstructSingleKeyCol<int32_t>(&multiKeys, outDataElementCount,
                                                 occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_LONG:
            {
                ReconstructSingleKeyCol<int64_t>(&multiKeys, outDataElementCount,
                                                 occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_FLOAT:
            {
                ReconstructSingleKeyCol<float>(&multiKeys, outDataElementCount, occupancyMask.get(),
                                               keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_DOUBLE:
            {
                ReconstructSingleKeyCol<double>(&multiKeys, outDataElementCount,
                                                occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_STRING:
            {
                ReconstructSingleKeyCol<std::string>(&multiKeys, outDataElementCount,
                                                     occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_INT8_T:
            {
                ReconstructSingleKeyCol<int8_t>(&multiKeys, outDataElementCount,
                                                occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            default:
                break;
            }
        }

        // Merge multipied arrays (values and occurrences)
        std::tuple<cuda_ptr<V>, cuda_ptr<int64_t>> mergedArrays =
            MergeMultipliedArrays<AGG, V, USE_VALUES, USE_KEY_OCCURRENCES>(values_, keyOccurrenceCount_,
                                                                           occupancyMask.get(), maxHashCount_,
                                                                           GB_ARRAY_MULTIPLIER);
        cuda_ptr<V> mergedValues = std::move(std::get<0>(mergedArrays));
        cuda_ptr<int64_t> mergedOccurrences = std::move(std::get<1>(mergedArrays));

        if (USE_VALUES)
        {
            GPUReconstruct::reconstructCol(values, outDataElementCount, mergedValues.get(),
                                           occupancyMask.get(), maxHashCount_);
        }
        if (USE_KEY_OCCURRENCES)
        {
            GPUReconstruct::reconstructCol(occurrences, outDataElementCount, mergedOccurrences.get(),
                                           occupancyMask.get(), maxHashCount_);
        }
    }

    /// Get the final results of GROUP BY operation on single GPU (keeps result on GPU)
    /// <param name="outKeysVector">pointer to empty CPU vector of GPU pointers (will be filled with
    ///   final key cols)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated
    ///   and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer (will be filled with count
    ///   of reconstructed elements)</param>
    /// CLEANUP: free all GPU pointers from outKeysVector by calling function FreeKeysVector
    void GetResults(std::vector<void*>* outKeysVector, O** outValues, int32_t* outDataElementCount)
    {
        static_assert(!std::is_same<AGG, AggregationFunctions::count>::value || std::is_same<O, int64_t>::value,
                      "GroupBy COUNT ouput data type O must be int64_t");
        Context& context = Context::getInstance();

        // Compute key occupancy mask
        cuda_ptr<int8_t> occupancyMask(maxHashCount_);
        kernel_source_indices_to_mask<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
            occupancyMask.get(), sourceIndices_, maxHashCount_);

        // Copy data types back from GPU
        std::unique_ptr<DataType[]> keyTypesHost = std::make_unique<DataType[]>(keysColCount_);
        GPUMemory::copyDeviceToHost(keyTypesHost.get(), keyTypes_, keysColCount_);

        // Reconstruct keys from all collected cols
        for (int32_t t = 0; t < keysColCount_; t++)
        {
            switch (keyTypesHost[t])
            {
            case DataType::COLUMN_INT:
            {
                ReconstructSingleKeyColKeep<int32_t>(outKeysVector, outDataElementCount,
                                                     occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_LONG:
            {
                ReconstructSingleKeyColKeep<int64_t>(outKeysVector, outDataElementCount,
                                                     occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_FLOAT:
            {
                ReconstructSingleKeyColKeep<float>(outKeysVector, outDataElementCount,
                                                   occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_DOUBLE:
            {
                ReconstructSingleKeyColKeep<double>(outKeysVector, outDataElementCount,
                                                    occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_STRING:
            {
                ReconstructSingleKeyColKeep<std::string>(outKeysVector, outDataElementCount,
                                                         occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            case DataType::COLUMN_INT8_T:
            {
                ReconstructSingleKeyColKeep<int8_t>(outKeysVector, outDataElementCount,
                                                    occupancyMask.get(), keysBuffer_ + t, maxHashCount_);
                break;
            }
            default:
                break;
            }
        }

        // Merge multipied arrays (values and occurrences)
        std::tuple<cuda_ptr<V>, cuda_ptr<int64_t>> mergedArrays =
            MergeMultipliedArrays<AGG, V, USE_VALUES, USE_KEY_OCCURRENCES>(values_, keyOccurrenceCount_,
                                                                           occupancyMask.get(), maxHashCount_,
                                                                           GB_ARRAY_MULTIPLIER);
        cuda_ptr<V> mergedValues = std::move(std::get<0>(mergedArrays));
        cuda_ptr<int64_t> mergedOccurrences = std::move(std::get<1>(mergedArrays));

        // Reconstruct aggregated values
        if (DIRECT_VALUES) // for min, max and sum: mergedValues.get() are direct results, just reconstruct them
        {
            if (!std::is_same<O, V>::value)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR, "Input and output value data type must "
                                                           "be the same in GROUP BY");
            }
            // reinterpret_cast is needed to solve compilation error
            GPUReconstruct::reconstructColKeep(outValues, outDataElementCount,
                                               reinterpret_cast<O*>(mergedValues.get()),
                                               occupancyMask.get(), maxHashCount_);
        }
        else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg: mergedValues.get() need to be divided by keyOccurrences_ and reconstructed
        {
            cuda_ptr<O> outValuesGPU(maxHashCount_);
            // Divide by counts to get averages for buckets
            try
            {
                GPUArithmetic::colCol<ArithmeticOperations::div>(outValuesGPU.get(), mergedValues.get(),
                                                                 mergedOccurrences.get(), maxHashCount_);
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
                                               occupancyMask.get(), maxHashCount_);
        }
        else // for count: reconstruct and return keyOccurrences_
        {
            if (!std::is_same<O, int64_t>::value)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR, "Output value data type in GROUP BY "
                                                           "with COUNT must be int64_t");
            }
            // reinterpret_cast is needed to solve compilation error
            // not reinterpreting anything here actually, outValues is int64_t** always in this else-branch
            GPUReconstruct::reconstructColKeep(reinterpret_cast<int64_t**>(outValues), outDataElementCount,
                                               mergedOccurrences.get(), occupancyMask.get(), maxHashCount_);
        }
    }


    /// Merge results from all devices and store to buffers on default device (multi GPU function)
    /// <param name="outKeysVector">pointer to empty CPU vector of GPU pointers
    ///   (will be filled with final key cols)</param>
    /// <param name="outValues">double pointer of output GPU buffer (will be allocated
    ///   and filled with final values)</param>
    /// <param name="outDataElementCount">output CPU buffer
    ///   (will be filled with count of reconstructed elements)</param>
    /// <param name="tables">vector of unique pointers
    ///   to IGroupBy objects with hash tables on every device (GPU)</param>
    void GetResults(std::vector<void*>* outKeysVector,
                    O** outValues,
                    int32_t* outDataElementCount,
                    std::vector<std::unique_ptr<IGroupBy>>& tables)
    {
        if (tables.size() <= 0) // invalid count of tables
        {
            CheckQueryEngineError(GPU_EXTENSION_ERROR, "Number of tables have to be at least 1.");
        }
        else if (tables.size() == 1 || tables[1].get() == nullptr) // just one table
        {
            GetResults(outKeysVector, outValues, outDataElementCount);
        }
        else // more tables
        {
            int32_t oldDeviceId = Context::getInstance().getBoundDeviceID();

            std::vector<void*> multiKeysAllHost; // this vector is oriented orthogonally to others (vector of cols)
            std::vector<V> valuesAllHost;
            std::vector<int64_t> occurrencesAllHost;
            int32_t sumElementCount = 0;

            // Copy data types back from GPU
            std::vector<DataType> keyTypesHost;
            keyTypesHost.resize(keysColCount_);
            GPUMemory::copyDeviceToHost(keyTypesHost.data(), keyTypes_, keysColCount_);

            // Pre-allocate pointer in multiKeysAllHost for max possible size
            size_t hashTablesSizesSum = maxHashCount_ * tables.size();
            for (int32_t t = 0; t < keysColCount_; t++)
            {
                switch (keyTypesHost[t])
                {
                case DataType::COLUMN_INT:
                {
                    multiKeysAllHost.emplace_back(new int32_t[hashTablesSizesSum]);
                    break;
                }
                case DataType::COLUMN_LONG:
                {
                    multiKeysAllHost.emplace_back(new int64_t[hashTablesSizesSum]);
                    break;
                }
                case DataType::COLUMN_FLOAT:
                {
                    multiKeysAllHost.emplace_back(new float[hashTablesSizesSum]);
                    break;
                }
                case DataType::COLUMN_DOUBLE:
                {
                    multiKeysAllHost.emplace_back(new double[hashTablesSizesSum]);
                    break;
                }
                case DataType::COLUMN_STRING:
                {
                    multiKeysAllHost.emplace_back(new CPUString());
                    break;
                }
                case DataType::COLUMN_INT8_T:
                {
                    multiKeysAllHost.emplace_back(new int8_t[hashTablesSizesSum]);
                    break;
                }
                default:
                    break;
                }
            }

            // Collect data from all devices (graphic cards) to host
            for (int32_t i = 0; i < tables.size(); i++)
            {
                if (tables[i].get() == nullptr)
                {
                    break;
                }
                // TODO change to cudaMemcpyPeerAsync
                GPUGroupBy<AGG, O, std::vector<void*>, V>* table =
                    reinterpret_cast<GPUGroupBy<AGG, O, std::vector<void*>, V>*>(tables[i].get());
                std::vector<void*> multiKeys;
                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->GetMaxHashCount());
                std::unique_ptr<int64_t[]> occurrences =
                    std::make_unique<int64_t[]>(table->GetMaxHashCount());
                int32_t elementCount;

                Context::getInstance().bindDeviceToContext(i);
                // Reconstruct multi-keys, values and occurrences
                table->ReconstructRawNumbers(multiKeys, values.get(), occurrences.get(), &elementCount);

                // Collect reconstructed raw numbers to *allHost buffers
                for (int32_t t = 0; t < keysColCount_; t++)
                {
                    switch (keyTypesHost[t])
                    {
                    case DataType::COLUMN_INT:
                    {
                        memcpy(reinterpret_cast<int32_t*>(multiKeysAllHost[t]) + sumElementCount,
                               multiKeys[t], elementCount * sizeof(int32_t));
                        delete[] reinterpret_cast<int32_t*>(multiKeys[t]);
                        break;
                    }
                    case DataType::COLUMN_LONG:
                    {
                        memcpy(reinterpret_cast<int64_t*>(multiKeysAllHost[t]) + sumElementCount,
                               multiKeys[t], elementCount * sizeof(int64_t));
                        delete[] reinterpret_cast<int64_t*>(multiKeys[t]);
                        break;
                    }
                    case DataType::COLUMN_FLOAT:
                    {
                        memcpy(reinterpret_cast<float*>(multiKeysAllHost[t]) + sumElementCount,
                               multiKeys[t], elementCount * sizeof(float));
                        delete[] reinterpret_cast<float*>(multiKeys[t]);
                        break;
                    }
                    case DataType::COLUMN_DOUBLE:
                    {
                        memcpy(reinterpret_cast<double*>(multiKeysAllHost[t]) + sumElementCount,
                               multiKeys[t], elementCount * sizeof(double));
                        delete[] reinterpret_cast<double*>(multiKeys[t]);
                        break;
                    }
                    case DataType::COLUMN_STRING:
                    {
                        CPUString* oldBuffer = reinterpret_cast<CPUString*>(multiKeysAllHost[t]);
                        CPUString* addBuffer = reinterpret_cast<CPUString*>(multiKeys[t]);
                        CPUString* newBuffer = new CPUString();
                        newBuffer->stringLengths.insert(newBuffer->stringLengths.end(),
                                                        oldBuffer->stringLengths.begin(),
                                                        oldBuffer->stringLengths.end());
                        newBuffer->allChars.insert(newBuffer->allChars.end(), oldBuffer->allChars.begin(),
                                                   oldBuffer->allChars.end());
                        newBuffer->stringLengths.insert(newBuffer->stringLengths.end(),
                                                        addBuffer->stringLengths.begin(),
                                                        addBuffer->stringLengths.end());
                        newBuffer->allChars.insert(newBuffer->allChars.end(), addBuffer->allChars.begin(),
                                                   addBuffer->allChars.end());
                        multiKeysAllHost[t] = newBuffer;
                        delete oldBuffer;
                        delete addBuffer;
                        break;
                    }
                    case DataType::COLUMN_INT8_T:
                    {
                        memcpy(reinterpret_cast<int8_t*>(multiKeysAllHost[t]) + sumElementCount,
                               multiKeys[t], elementCount * sizeof(int8_t));
                        delete[] reinterpret_cast<int8_t*>(multiKeys[t]);
                        break;
                    }
                    default:
                        break;
                    }
                }
                if (USE_VALUES)
                {
                    valuesAllHost.insert(valuesAllHost.end(), values.get(), values.get() + elementCount);
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
                void** multiKeysAllGPU;
                std::vector<void*> hostPointersToKeysAll;
                AllocKeysBuffer(&multiKeysAllGPU, keyTypesHost, sumElementCount, &hostPointersToKeysAll);
                cuda_ptr<V> valuesAllGPU(sumElementCount);
                cuda_ptr<int64_t> occurrencesAllGPU(sumElementCount);

                // Copy collected data to one GPU
                for (int32_t t = 0; t < keysColCount_; t++)
                {
                    switch (keyTypesHost[t])
                    {
                    case DataType::COLUMN_INT:
                    {
                        GPUMemory::copyHostToDevice(reinterpret_cast<int32_t*>(hostPointersToKeysAll[t]),
                                                    reinterpret_cast<int32_t*>(multiKeysAllHost[t]),
                                                    sumElementCount);
                        delete[] reinterpret_cast<int32_t*>(multiKeysAllHost[t]);
                        break;
                    }
                    case DataType::COLUMN_LONG:
                    {
                        GPUMemory::copyHostToDevice(reinterpret_cast<int64_t*>(hostPointersToKeysAll[t]),
                                                    reinterpret_cast<int64_t*>(multiKeysAllHost[t]),
                                                    sumElementCount);
                        delete[] reinterpret_cast<int64_t*>(multiKeysAllHost[t]);
                        break;
                    }
                    case DataType::COLUMN_FLOAT:
                    {
                        GPUMemory::copyHostToDevice(reinterpret_cast<float*>(hostPointersToKeysAll[t]),
                                                    reinterpret_cast<float*>(multiKeysAllHost[t]),
                                                    sumElementCount);
                        delete[] reinterpret_cast<float*>(multiKeysAllHost[t]);
                        break;
                    }
                    case DataType::COLUMN_DOUBLE:
                    {
                        GPUMemory::copyHostToDevice(reinterpret_cast<double*>(hostPointersToKeysAll[t]),
                                                    reinterpret_cast<double*>(multiKeysAllHost[t]),
                                                    sumElementCount);
                        delete[] reinterpret_cast<double*>(multiKeysAllHost[t]);
                        break;
                    }
                    case DataType::COLUMN_STRING:
                    {
                        CPUString* strKeys = reinterpret_cast<CPUString*>(multiKeysAllHost[t]);
                        cuda_ptr<int32_t> keysAllGPUStringLengths(sumElementCount);
                        GPUMemory::copyHostToDevice(keysAllGPUStringLengths.get(),
                                                    strKeys->stringLengths.data(), sumElementCount);

                        // Construct new GPUString with all keys (might be duplicated yet)
                        GPUMemory::GPUString keysAllGPU;
                        GPUMemory::alloc(&(keysAllGPU.stringIndices), sumElementCount);
                        GPUReconstruct::PrefixSum(keysAllGPU.stringIndices,
                                                  keysAllGPUStringLengths.get(), sumElementCount);
                        GPUMemory::alloc(&(keysAllGPU.allChars), strKeys->allChars.size());
                        GPUMemory::copyHostToDevice(keysAllGPU.allChars, strKeys->allChars.data(),
                                                    strKeys->allChars.size());

                        // Copy struct itself
                        GPUMemory::copyHostToDevice(reinterpret_cast<GPUMemory::GPUString*>(hostPointersToKeysAll[t]),
                                                    &keysAllGPU, 1);
                        delete strKeys;
                        break;
                    }
                    case DataType::COLUMN_INT8_T:
                    {
                        GPUMemory::copyHostToDevice(reinterpret_cast<int8_t*>(hostPointersToKeysAll[t]),
                                                    reinterpret_cast<int8_t*>(multiKeysAllHost[t]),
                                                    sumElementCount);
                        delete[] reinterpret_cast<int8_t*>(multiKeysAllHost[t]);
                        break;
                    }
                    default:
                        break;
                    }
                }
                if (USE_VALUES)
                {
                    GPUMemory::copyHostToDevice(valuesAllGPU.get(), valuesAllHost.data(), sumElementCount);
                }
                if (USE_KEY_OCCURRENCES)
                {
                    GPUMemory::copyHostToDevice(occurrencesAllGPU.get(), occurrencesAllHost.data(), sumElementCount);
                }

                // Merge results
                if (DIRECT_VALUES) // for min, max and sum
                {
                    GPUGroupBy<AGG, O, std::vector<void*>, V> finalGroupBy(sumElementCount, keyTypesHost);
                    finalGroupBy.GroupBy(hostPointersToKeysAll, valuesAllGPU.get(), sumElementCount);
                    finalGroupBy.GetResults(outKeysVector, outValues, outDataElementCount);
                }
                else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg
                {
                    V* valuesMerged = nullptr;
                    int64_t* occurencesMerged = nullptr;

                    // TODO after implementation of multi-value GroupBy use it here
                    // Calculate sum of values
                    // Initialize new empty sumGroupBy table
                    std::vector<void*> keysToDiscard;
                    GPUGroupBy<AggregationFunctions::sum, V, std::vector<void*>, V> sumGroupBy(sumElementCount, keyTypesHost);
                    sumGroupBy.GroupBy(hostPointersToKeysAll, valuesAllGPU.get(), sumElementCount);
                    sumGroupBy.GetResults(&keysToDiscard, &valuesMerged, outDataElementCount);
                    FreeKeysVector(keysToDiscard, keyTypesHost);

                    // Calculate sum of occurences
                    // Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::vector<void*>, int64_t> countGroupBy(
                        sumElementCount, keyTypesHost, sumGroupBy.sourceIndices_, sumGroupBy.keysBuffer_);
                    countGroupBy.GroupBy(hostPointersToKeysAll, occurrencesAllGPU.get(), sumElementCount);
                    countGroupBy.GetResults(outKeysVector, &occurencesMerged, outDataElementCount);

                    // Divide merged values by merged occurences to get final averages
                    GPUMemory::alloc(outValues, *outDataElementCount);
                    GPUArithmetic::colCol<ArithmeticOperations::div>(*outValues, valuesMerged, occurencesMerged,
                                                                     *outDataElementCount);

                    GPUMemory::free(valuesMerged);
                    GPUMemory::free(occurencesMerged);
                }
                else // for count
                {
                    if (!std::is_same<O, int64_t>::value)
                    {
                        CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                              "Output value data type in GROUP BY with COUNT must "
                                              "be int64_t");
                    }
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::vector<void*>, int64_t> finalGroupBy(
                        sumElementCount, keyTypesHost);
                    finalGroupBy.GroupBy(hostPointersToKeysAll, occurrencesAllGPU.get(), sumElementCount);
                    // reinterpret_cast is needed to solve compilation error
                    finalGroupBy.GetResults(outKeysVector, reinterpret_cast<int64_t**>(outValues),
                                            outDataElementCount);
                }

                // TODO free everything (check in code) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                FreeKeysBuffer(multiKeysAllGPU, keyTypes_, keysColCount_);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};
