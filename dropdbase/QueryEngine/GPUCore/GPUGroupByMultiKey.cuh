#pragma once

#include "../../DataType.h"
#include "GPUGroupByString.cuh"


__device__ int32_t GetHash(DataType* keysTypes, int32_t keysColCount, void* keys);


/// GROUP BY Kernel processes input (inKeys and inValues). New keys from inKeys are added
/// to the hash table and values from inValues are aggregated.
template <typename AGG, typename V>
__global__ void kernel_group_by_multi_key(DataType* keysTypes,
                                          int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          V* values,
                                          int64_t* keyOccurenceCount,
                                          int32_t maxHashCount,
                                          void** inKeys,
                                          V* inValues,
                                          int32_t dataElementCount,
                                          int32_t* errorFlag)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Calculate hash
        //const int32_t hash = GetHash(keysTypes, keysColCount, ...);

        
        // TODO

    }
}


/// GROUP BY class for multi-keys, for MIN, MAX and SUM.
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::vector<std::pair<void*, DataType>>, V> : public IGroupBy
{
public:
    /// Indices to input keys - because of atomicity
    int32_t* sourceIndices_ = nullptr;
    /// Types of keys
    DataType* keysTypes_ = nullptr;
    /// Count of key columns
    const int32_t keysColCount_;
    /// Keys buffer - all found combination of keys are stored here
    void** keysBuffer_ = nullptr;

private:
    /// Value buffer of the hash table
    V* values_ = nullptr;
    /// Count of values aggregated per key (helper buffer of the hash table)
    int64_t* keyOccurenceCount_ = nullptr;

    /// Size of the hash table (max. count of unique keys)
    const int32_t maxHashCount_;
    /// Error flag swapper for error checking after kernel runs
    ErrorFlagSwapper errorFlagSwapper_;

public:
    /// Create GPUGroupBy object and allocate a hash table (buffers for key, values and key occurence counts)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    GPUGroupBy(int32_t maxHashCount, const std::vector<DataType> keyTypes)
    : maxHashCount_(maxHashCount), keysColCount_(keyTypes.size())
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            // TODO alloc keysTypes_
            // TODO alloc keysBuffer_

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
            // TODO for key : keysBuffer_
            if (keysTypes_)
            {
                GPUMemory::free(keysTypes_);
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
        // TODO fill keysTypes_
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="sourceIndices">GPU buffer with existing sourceIndices (will be copied to a new buffer)</param>
    /// <param name="keysTypes">key column types (will be copied to a new buffer)</param>
    /// <param name="keysColCount">count of key columns</param>
    /// <param name="keysBuffer">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, DataType* keysTypes, int32_t keysColCount, void** keysBuffer)
    : maxHashCount_(maxHashCount)
    {
        // TODO
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        // TODO for key : keysBuffer_
        GPUMemory::free(values_);
        GPUMemory::free(keyOccurenceCount_);
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffers with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void groupBy(std::vector<void*> inKeys, V* inValues, int32_t dataElementCount)
    {

        if (dataElementCount > 0)
        {
            kernel_group_by_multi_key<AGG>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    keys_, values_, keyOccurenceCount_, maxHashCount_, inKeys, inValues,
                    dataElementCount, errorFlagSwapper_.GetFlagPointer());
            errorFlagSwapper_.Swap();
        }
    }
}
