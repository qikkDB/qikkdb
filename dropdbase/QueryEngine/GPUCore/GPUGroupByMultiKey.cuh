#pragma once

#include "../../DataType.h"
#include "../GPUError.h"
#include "GPUGroupByString.cuh"


__device__ int32_t GetHash(DataType* keyTypes, int32_t keysColCount, void** inKeys, int32_t i);

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

/// GROUP BY Kernel processes input (inKeys and inValues). New keys from inKeys are added
/// to the hash table and values from inValues are aggregated.
template <typename AGG, typename V>
__global__ void kernel_group_by_multi_key(DataType* keyTypes,
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
        const int32_t hash = GetHash(keyTypes, keysColCount, inKeys, i);

        int32_t foundIndex = -1;
        for (int32_t j = 0; j < maxHashCount; j++)
        {
            // Calculate index to hash-table from hash
            const int32_t index = abs((hash + j) % maxHashCount);

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
            AGG{}(&values[foundIndex], inValues[i]);
            atomicAdd(reinterpret_cast<cuUInt64*>(&keyOccurenceCount[foundIndex]), 1);
        }
    }
}

__global__ void kernel_collect_string_lengths(int32_t* stringLengths, int32_t* sourceIndices,
    GPUMemory::GPUString ** inKeysSingleCol, GPUMemory::GPUString ** keysBufferSingleCol, int32_t maxHashCount);

__global__ void kernel_collect_multi_keys(DataType* keyTypes,
                                          int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          int32_t** stringLengthsBuffers,
                                          int32_t maxHashCount,
                                          void** inKeys);


/// GROUP BY class for multi-keys, for MIN, MAX and SUM.
template <typename AGG, typename O, typename V>
class GPUGroupBy<AGG, O, std::vector<void*>, V> : public IGroupBy
{
public:
    /// Indices to input keys - because of atomicity
    int32_t* sourceIndices_ = nullptr;
    /// Types of keys
    DataType* keyTypes_ = nullptr;
    /// Count of key columns
    const int32_t keysColCount_;
    /// Keys buffer - all found combination of keys are stored here
    void** keysBuffer_ = nullptr;

    std::vector<int32_t> stringKeyColIds_;

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
    GPUGroupBy(int32_t maxHashCount, std::vector<DataType> keyTypes)
    : maxHashCount_(maxHashCount), keysColCount_(keyTypes.size())
    {
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&keyTypes_, keysColCount_);
            GPUMemory::alloc(&keysBuffer_, keysColCount_);
            for (int32_t i = 0; i < keysColCount_; i++)
            {
                switch (keyTypes[i])
                {
                case DataType::COLUMN_INT:
                {
                    int32_t * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, maxHashCount_);
                    GPUMemory::copyHostToDevice(reinterpret_cast<int32_t**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                case DataType::COLUMN_LONG:
                {
                    int64_t * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, maxHashCount_);
                    GPUMemory::copyHostToDevice(reinterpret_cast<int64_t**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                case DataType::COLUMN_FLOAT:
                {
                    float * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, maxHashCount_);
                    GPUMemory::copyHostToDevice(reinterpret_cast<float**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                case DataType::COLUMN_DOUBLE:
                {
                    double * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, maxHashCount_);
                    GPUMemory::copyHostToDevice(reinterpret_cast<double**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                case DataType::COLUMN_STRING:
                {
                    stringKeyColIds_.emplace_back(i);
                    GPUMemory::GPUString * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, 1);
                    GPUMemory::copyHostToDevice(reinterpret_cast<GPUMemory::GPUString**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                case DataType::COLUMN_INT8_T:
                {
                    int8_t * gpuKeyCol;
                    GPUMemory::alloc(&gpuKeyCol, maxHashCount_);
                    GPUMemory::copyHostToDevice(reinterpret_cast<int8_t**>(keysBuffer_ + i), &gpuKeyCol, 1);
                    break;
                }
                default:
                    CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                          "Multi-key GROUP BY with keys of type " +
                                              std::to_string(keyTypes[i]) + " is not supported");
                    break;
                }
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
            if (keyTypes_)
            {
                GPUMemory::free(keyTypes_);
            }
            if (keysBuffer_)
            {
                for (int32_t i = 0; i < keysColCount_; i++)
                {
                    void * ptr;
                    GPUMemory::copyDeviceToHost(&ptr, keysBuffer_+i, 1);
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
            if (keyOccurenceCount_)
            {
                GPUMemory::free(keyOccurenceCount_);
            }
            throw;
        }
        GPUMemory::fillArray(sourceIndices_, GBS_SOURCE_INDEX_EMPTY_KEY, maxHashCount_);
        GPUMemory::fillArray(values_, AGG::template getInitValue<V>(), maxHashCount_);
        GPUMemory::copyHostToDevice(keyTypes_, keyTypes.data(), keysColCount_);
    }

    /// Create GPUGroupBy object with existing keys (allocate whole new hash table)
    /// <param name="maxHashCount">size of the hash table (max. count of unique keys)</param>
    /// <param name="sourceIndices">GPU buffer with existing sourceIndices (will be copied to a new buffer)</param>
    /// <param name="keyTypes">key column types (will be copied to a new buffer)</param>
    /// <param name="keysColCount">count of key columns</param>
    /// <param name="keysBuffer">GPU buffer with existing keys (will be copied to a new buffer)</param>
    GPUGroupBy(int32_t maxHashCount, int32_t* sourceIndices, DataType* keyTypes, int32_t keysColCount, void** keysBuffer)
    : maxHashCount_(maxHashCount)
    {
        // TODO
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        GPUMemory::free(keyTypes_);
        for (int32_t i = 0; i < keysColCount_; i++)
        {
                void * ptr;
                GPUMemory::copyDeviceToHost(&ptr, keysBuffer_+i, 1);
                if (ptr)
                {
                    GPUMemory::free(ptr);
                }
        }
        GPUMemory::free(keysBuffer_);
        GPUMemory::free(values_);
        GPUMemory::free(keyOccurenceCount_);
    }

    GPUGroupBy(const GPUGroupBy&) = delete;
    GPUGroupBy& operator=(const GPUGroupBy&) = delete;


    /// Run GROUP BY on one input buffer - callable repeatedly on the blocks of the input columns
    /// <param name="inKeys">input buffers with keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    void groupBy(std::vector<void*> inKeysVector, V* inValues, int32_t dataElementCount)
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

            // Run group by kernel (get sourceIndices and aggregate values)
            kernel_group_by_multi_key<AGG>
                <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                    keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, values_, keyOccurenceCount_,
                    maxHashCount_, inKeys.get(), inValues, dataElementCount, errorFlagSwapper_.GetFlagPointer());
            errorFlagSwapper_.Swap();

            cuda_ptr<int32_t*> stringLengthsBuffers(keysColCount_, 0);  // alloc pointers and set to nullptr
            for(int32_t t : stringKeyColIds_)
            {
                int32_t * stringLengths;
                GPUMemory::alloc(&stringLengths, maxHashCount_);
                GPUMemory::copyHostToDevice(stringLengthsBuffers.get() + t, &stringLengths, 1); // copy pointer to stringLengths
                kernel_collect_string_lengths<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                    stringLengths, sourceIndices_,
                    reinterpret_cast<GPUMemory::GPUString **>(inKeys.get() + t),
                    reinterpret_cast<GPUMemory::GPUString **>(keysBuffer_ + t), maxHashCount_);
            }

            // Collect multi-keys from inKeys according to sourceIndices
            kernel_collect_multi_keys<<<context.calcGridDim(maxHashCount_), context.getBlockDim()>>>(
                keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, stringLengthsBuffers.get(), maxHashCount_,
                inKeys.get());

            CheckCudaError(cudaGetLastError());
        }
    }


    /// Get the size of the hash table (max. count of unique multi-keys)
    /// <returns>size of the hash table</returns>
    int32_t getMaxHashCount()
    {
        return maxHashCount_;
    }


    /// Get the final results of GROUP BY operation - for operations Min, Max and Sum - on single
    /// GPU <param name="outKeys">pointer to GPUString struct (will be allocated and filled with
    /// final keys)</param> <param name="outValues">double pointer of output GPU buffer (will be
    /// allocated and filled with final values)</param> <param name="outDataElementCount">output CPU
    /// buffer (will be filled with count of reconstructed elements)</param>
    void getResults(std::vector<void*>* outKeysVector, O** outValues, int32_t* outDataElementCount)
    {
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
                int32_t* keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<int32_t**>(keysBuffer_+t), 1);
                int32_t* outKeysSingleCol;
                GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount,
                                                   keyBufferSingleCol,
                                                   occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            case DataType::COLUMN_LONG:
            {
                int64_t* keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<int64_t**>(keysBuffer_+t), 1);
                int64_t* outKeysSingleCol;
                GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount,
                                                   keyBufferSingleCol,
                                                   occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            case DataType::COLUMN_FLOAT:
            {
                float* keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<float**>(keysBuffer_+t), 1);
                float* outKeysSingleCol;
                GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount,
                                                   keyBufferSingleCol,
                                                   occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            case DataType::COLUMN_DOUBLE:
            {
                double* keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<double**>(keysBuffer_+t), 1);
                double* outKeysSingleCol;
                GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount,
                                                   keyBufferSingleCol,
                                                   occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            case DataType::COLUMN_STRING:
            {
                // Copy struct (not pointer to struct)
                GPUMemory::GPUString keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<GPUMemory::GPUString*>(keysBuffer_[t]), 1);
                // Reconstruct string keys
                GPUMemory::GPUString * outKeysSingleCol = new GPUMemory::GPUString[1];
                GPUReconstruct::ReconstructStringColKeep(outKeysSingleCol, outDataElementCount, keyBufferSingleCol,
                    occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            case DataType::COLUMN_INT8_T:
            {
                int8_t* keyBufferSingleCol;
                GPUMemory::copyDeviceToHost(&keyBufferSingleCol, reinterpret_cast<int8_t**>(keysBuffer_+t), 1);
                int8_t* outKeysSingleCol;
                GPUReconstruct::reconstructColKeep(&outKeysSingleCol, outDataElementCount,
                                                   keyBufferSingleCol,
                                                   occupancyMask.get(), maxHashCount_);
                outKeysVector->emplace_back(outKeysSingleCol);
                break;
            }
            default:
                break;
            }
        }
        // Reconstruct aggregated values
        GPUReconstruct::reconstructColKeep(outValues, outDataElementCount, values_,
                                           occupancyMask.get(), maxHashCount_);
    }
};
