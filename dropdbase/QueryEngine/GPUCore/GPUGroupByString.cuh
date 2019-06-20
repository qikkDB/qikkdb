#pragma once

#include "GPUFilter.cuh"
#include "GPUGroupBy.cuh"
#include "GPUStringUnary.cuh"


constexpr int32_t SOURCE_INDEX_EMPTY_KEY = -1;
constexpr int32_t SOURCE_INDEX_KEY_IN_BUFFER = -2;
constexpr int32_t STRING_HASH_SEED = 31;


__device__ int32_t GetHash(char* text, int32_t length)
{
    int32_t hash = 0;
    for (int32_t i = 0; i < length; i++)
    {
        hash = STRING_HASH_SEED * hash + text[i];
    }
    return hash;
}


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
__global__ void group_by_kernel(int32_t* sourceIndices,
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
            printf("%d (%c%c%c...): %d\n", i, inKeyChars[0], inKeyChars[1], inKeyChars[2], index);
            // Check if key is not empty and key is not equal to the currently inserted key
            if ((sourceIndices[index] >= 0 &&
                 FilterConditions::notEqual{}.compareStrings( // find in inKeys at first
                     inKeyChars, inKeyLength,
                     inKeys.allChars + GetStringIndex(inKeys.stringIndices, sourceIndices[index]),
                     GetStringLength(inKeys.stringIndices, sourceIndices[index]))) ||
                (sourceIndices[index] == SOURCE_INDEX_KEY_IN_BUFFER &&
                 FilterConditions::notEqual{}.compareStrings( // and in keysBuffer next
                     inKeyChars, inKeyLength, keysBuffer.allChars + GetStringIndex(keysBuffer.stringIndices, index),
                     GetStringLength(keysBuffer.stringIndices, index))))
            {
                printf("%d (%c%c%c...): c1\n", i, inKeyChars[0], inKeyChars[1], inKeyChars[2]);
                continue;
            }

            // If key is empty
            if (sourceIndices[index] == SOURCE_INDEX_EMPTY_KEY)
            {
                // Compare key at index with Empty and if equals, store there inKey
                int32_t old = genericAtomicCAS<int32_t>(&sourceIndices[index], SOURCE_INDEX_EMPTY_KEY, i);

                // Check if some other thread stored different key to this index
                if (old != SOURCE_INDEX_EMPTY_KEY &&
                    (old != i && (sourceIndices[index] >= 0 &&
                                  FilterConditions::notEqual{}.compareStrings( // find in inKeys at first
                                      inKeyChars, inKeyLength,
                                      inKeys.allChars + GetStringIndex(inKeys.stringIndices, sourceIndices[index]),
                                      GetStringLength(inKeys.stringIndices, sourceIndices[index]))) ||
                     (sourceIndices[index] == SOURCE_INDEX_KEY_IN_BUFFER &&
                      FilterConditions::notEqual{}.compareStrings( // and in keysBuffer next
                          inKeyChars, inKeyLength, keysBuffer.allChars + GetStringIndex(keysBuffer.stringIndices, index),
                          GetStringLength(keysBuffer.stringIndices, index)))))
                {
					printf("%d (%c%c%c...): cA\n", i, inKeyChars[0], inKeyChars[1], inKeyChars[2]);
                    continue; // Try to find another index
                }
            }
            else if (sourceIndices[index] != i)
            {
                // Check string equality once again
                if ((sourceIndices[index] >= 0 &&
                     FilterConditions::notEqual{}.compareStrings( // find in inKeys at first
                         inKeyChars, inKeyLength,
                         inKeys.allChars + GetStringIndex(inKeys.stringIndices, sourceIndices[index]),
                         GetStringLength(inKeys.stringIndices, sourceIndices[index]))) ||
                    (sourceIndices[index] == SOURCE_INDEX_KEY_IN_BUFFER &&
                     FilterConditions::notEqual{}.compareStrings( // and in keysBuffer next
                         inKeyChars, inKeyLength, keysBuffer.allChars + GetStringIndex(keysBuffer.stringIndices, index),
                         GetStringLength(keysBuffer.stringIndices, index))))
                {
                    printf("%d (%c%c%c...): ce\n", i, inKeyChars[0], inKeyChars[1], inKeyChars[2]);
                    continue; // try to find another index
                }
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


/// GROUP BY generic class (for MIN, MAX and SUM).
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, GPUMemory::GPUString, V> : public IGroupBy
{
    // TODO private:
public:
    int32_t* sourceIndices_ = nullptr;
    int32_t* stringLengths_ = nullptr;
    GPUMemory::GPUString keysBuffer_;

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
            GPUMemory::alloc(&(keysBuffer_.stringIndices), maxHashCount_);
            keysBuffer_.allChars = nullptr;

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
            if (keysBuffer_.stringIndices)
            {
                GPUMemory::free(keysBuffer_.stringIndices);
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
        GPUMemory::fillArray(sourceIndices_, SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
    }


    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="keys">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, GPUMemory::GPUString keys) : maxHashCount_(maxHashCount)
    {
        // TODO !  <<<< <<<< <<<< <<<<
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
            group_by_kernel<AGG>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    sourceIndices_, stringLengths_, keysBuffer_, values_, keyOccurenceCount_, maxHashCount_,
                    inKeys, inValues, dataElementCount, errorFlagSwapper_.GetFlagPointer());
            errorFlagSwapper_.Swap();
        }
    }
};
