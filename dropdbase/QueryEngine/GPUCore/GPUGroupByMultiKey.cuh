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

__constant__ const uint32_t CRC_32_TAB[] =
    {/* CRC polynomial 0xedb88320 */
     0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535,
     0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd,
     0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d,
     0x6ddde4eb, 0xf4d4b551, 0x83d385c7, 0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
     0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4,
     0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
     0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59, 0x26d930ac,
     0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
     0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab,
     0xb6662d3d, 0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f,
     0x9fbfe4a5, 0xe8b8d433, 0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb,
     0x086d3d2d, 0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
     0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea,
     0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65, 0x4db26158, 0x3ab551ce,
     0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a,
     0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
     0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409,
     0xce61e49f, 0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
     0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739,
     0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
     0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1, 0xf00f9344, 0x8708a3d2, 0x1e01f268,
     0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0,
     0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8,
     0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
     0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef,
     0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703,
     0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7,
     0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d, 0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
     0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae,
     0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
     0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777, 0x88085ae6,
     0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
     0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d,
     0x3e6e77db, 0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5,
     0x47b2cf7f, 0x30b5ffe9, 0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605,
     0xcdd70693, 0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
     0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};


/// Compute hash of multi-key
__device__ int32_t GetHash(DataType* keyTypes, int32_t keysColCount, void** inKeys, int8_t** inKeysNullMask, int32_t i, int32_t hashCoef);

/// Chceck for equality of two multi-keys
__device__ bool AreEqualMultiKeys(DataType* keyTypes,
                                  int32_t keysColCount,
                                  void** keysA,
                                  int8_t** keysANullMask,
                                  int32_t indexA,
                                  void** keysB,
                                  int8_t** keysBNullMask,
                                  int32_t indexB);

/// Check if multi-key is new (not present in keys buffer nor in sourceIndices)
/// Input multi-key is defined by inKeys and i
/// <param name="sourceIndices">points to inKeys</param>
/// <param name="index">points to keysBuffer</param>
__device__ bool IsNewMultiKey(DataType* keyTypes,
                              int32_t keysColCount,
                              void** inKeys,
                              int8_t** inKeysNullMask,
                              int32_t i,
                              void** keysBuffer,
                              int8_t** keysNullBuffer,
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
                     int8_t*** keysNullBuffer,
                     std::vector<DataType> keyTypes,
                     int32_t rowCount,
                     std::vector<void*>* pointers = nullptr,
                     std::vector<int8_t*>* pointersNullMask = nullptr);

/// Free 2-dimensional buffer for multi-keys storage
void FreeKeysBuffer(void** keysBuffer, int8_t** keysNullBuffer, DataType* keyTypes, int32_t keysColCount);

/// Free buffers from vector
void FreeKeysVector(std::vector<void*> keysVector, std::vector<DataType> keyTypes);

/// GROUP BY Kernel processes input (inKeys and inValues). New keys from inKeys are added
/// to the hash table and values from inValues are aggregated.
template <typename AGG, typename V>
__global__ void kernel_group_by_multi_key(DataType* keyTypes,
                                          const int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          int8_t** keysNullBuffer,
                                          V* values,
                                          int8_t* valuesNullMask,
                                          int64_t* keyOccurrenceCount,
                                          const int32_t maxHashCount,
                                          void** inKeys,
                                          V* inValues,
                                          const int32_t dataElementCount,
                                          const int32_t arrayMultiplier,
                                          const int32_t hashCoef,
                                          int32_t* errorFlag,
                                          int8_t** inKeysNullMask,
                                          int8_t* inValuesNullMask)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Get bool if input value is NULL
        const int32_t bitMaskIdx = (i / (sizeof(int8_t) * 8));
        const int32_t shiftIdx = (i % (sizeof(int8_t) * 8));
        const bool nullValue =
            (inValuesNullMask != nullptr) && ((inValuesNullMask[bitMaskIdx] >> shiftIdx) & 1);
        int32_t foundIndex = -1;

        // Calculate hash
        const int32_t hash =
            abs(GetHash(keyTypes, keysColCount, inKeys, inKeysNullMask, i, hashCoef)) % maxHashCount;
        for (int32_t j = 0; j < maxHashCount; j++)
        {
            // Calculate index to hash-table from hash
            const int32_t index = (hash + j) % maxHashCount;

            // Check if key is not empty and key is not equal to the currently inserted key
            if (sourceIndices[index] != GBS_SOURCE_INDEX_EMPTY_KEY &&
                IsNewMultiKey(keyTypes, keysColCount, inKeys, inKeysNullMask, i, keysBuffer,
                              keysNullBuffer, sourceIndices, index))
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
                    IsNewMultiKey(keyTypes, keysColCount, inKeys, inKeysNullMask, i, keysBuffer,
                                  keysNullBuffer, sourceIndices, index))
                {
                    continue; // Try to find another index
                }
            }
            else if (sourceIndices[index] != i &&
                     IsNewMultiKey(keyTypes, keysColCount, inKeys, inKeysNullMask, i, keysBuffer,
                                   keysNullBuffer, sourceIndices, index))
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
            // If the value is not null, aggregate
            if (!nullValue)
            {
                if (values)
                {
                    // Aggregate value
                    AGG{}(values + foundIndex * arrayMultiplier + threadIdx.x % arrayMultiplier, inValues[i]);
                    if (valuesNullMask[foundIndex])
                    {
                        valuesNullMask[foundIndex] = 0;
                    }
                }
                if (keyOccurrenceCount)
                {
                    // Increment counter
                    atomicAdd(reinterpret_cast<cuUInt64*>(keyOccurrenceCount + foundIndex * arrayMultiplier +
                                                          threadIdx.x % arrayMultiplier),
                              1);
                }
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
                                          int8_t** keysNullBuffer,
                                          GPUMemory::GPUString* stringSideBuffers,
                                          int32_t** stringLengthsBuffers,
                                          int32_t maxHashCount,
                                          void** inKeys,
                                          int8_t** inKeysNullMask);


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
    int8_t** keysNullBuffer_ = nullptr; // wide, uncompressed

private:
    /// Types of keys
    DataType* keyTypes_ = nullptr;
    /// Count of key columns
    const int32_t keysColCount_;
    /// Indices of string key columns
    std::vector<int32_t> stringKeyColIds_;

    /// Value buffer of the hash table
    V* values_ = nullptr;
    int8_t* valuesNullMask_ = nullptr; // wide, uncompressed
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
        // TODO check if maxHashCount is integer pow of 2
        const size_t multipliedCount = static_cast<size_t>(maxHashCount_) * GB_ARRAY_MULTIPLIER;
        try
        {
            // Allocate buffers needed for key storing
            GPUMemory::alloc(&sourceIndices_, maxHashCount_);
            GPUMemory::alloc(&keyTypes_, keysColCount_);
            AllocKeysBuffer(&keysBuffer_, &keysNullBuffer_, keyTypes, maxHashCount_);
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
                GPUMemory::allocAndSet(&valuesNullMask_, 1, maxHashCount_);
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
            if (keysNullBuffer_)
            {
                for (int32_t i = 0; i < keysColCount_; i++)
                {
                    int8_t* ptr;
                    GPUMemory::copyDeviceToHost(&ptr, keysNullBuffer_ + i, 1);
                    if (ptr)
                    {
                        GPUMemory::free(ptr);
                    }
                }
                GPUMemory::free(keysNullBuffer_);
            }
            if (values_)
            {
                GPUMemory::free(values_);
            }
            if (valuesNullMask_)
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
    GPUGroupBy(int32_t maxHashCount, std::vector<DataType> keyTypes, int32_t* sourceIndices, void** keysBuffer, int8_t** keysNullBuffer)
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

            int8_t* myNullMask;
            GPUMemory::copyDeviceToHost(&myNullMask, keysNullBuffer_ + i, 1);
            int8_t* srcNullMask;
            GPUMemory::copyDeviceToHost(&srcNullMask, keysNullBuffer + i, 1);
            GPUMemory::copyDeviceToDevice(myNullMask, srcNullMask, maxHashCount_);
        }
    }

    ~GPUGroupBy()
    {
        GPUMemory::free(sourceIndices_);
        FreeKeysBuffer(keysBuffer_, keysNullBuffer_, keyTypes_, keysColCount_);
        GPUMemory::free(keyTypes_);
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
    /// <param name="inKeysVector">input vector of buffers with keys</param>
    /// <param name="inKeysNullMaskVector">input vector of null masks of keys</param>
    /// <param name="inValues">input buffer with values</param>
    /// <param name="dataElementCount">row count to process</param>
    /// <param name="inValuesNullMask">null mask of values</param>
    void ProcessBlock(std::vector<void*> inKeysVector,
                      std::vector<int8_t*> inKeysNullMaskVector,
                      V* inValues,
                      int32_t dataElementCount,
                      int8_t* inValuesNullMask = nullptr)
    {
        if (dataElementCount > 0)
        {
            if (inKeysVector.size() != keysColCount_)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                      "Incorrect number of key columns in ProcessBlock");
            }
            if (inKeysNullMaskVector.size() != keysColCount_)
            {
                CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                      "Incorrect number of key null masks in ProcessBlock");
            }
            Context& context = Context::getInstance();
            // Convert vector to GPU void
            cuda_ptr<void*> inKeys(keysColCount_);
            GPUMemory::copyHostToDevice(inKeys.get(), inKeysVector.data(), keysColCount_);
            cuda_ptr<int8_t*> inKeysNullMasks(keysColCount_);
            GPUMemory::copyHostToDevice(inKeysNullMasks.get(), inKeysNullMaskVector.data(), keysColCount_);

            // Run group by kernel (get sourceIndices and aggregate values).
            // Parameter hashCoef is comptued as n-th root of maxHashCount, where n is a number of key columns
            kernel_group_by_multi_key<AGG><<<context.calcGridDim(dataElementCount), 768>>>(
                keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, keysNullBuffer_, values_,
                valuesNullMask_, keyOccurrenceCount_, maxHashCount_, inKeys.get(), inValues,
                dataElementCount, GB_ARRAY_MULTIPLIER, static_cast<int32_t>(log2f(maxHashCount_) + 0.5f),
                errorFlagSwapper_.GetFlagPointer(), inKeysNullMasks.get(), inValuesNullMask);
            errorFlagSwapper_.Swap();
            cudaDeviceSynchronize();
            CheckCudaError(cudaGetLastError());

            cuda_ptr<int32_t*> stringLengthsBuffers(keysColCount_, 0); // alloc pointers and set to nullptr
            cuda_ptr<GPUMemory::GPUString> stringSideBuffers(keysColCount_, 0); // alloc clean structs on gpu

            // Copy strings from keys buffer to side buffer
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
                keyTypes_, keysColCount_, sourceIndices_, keysBuffer_, keysNullBuffer_,
                stringSideBuffers.get(), stringLengthsBuffers.get(), maxHashCount_, inKeys.get(),
                inKeysNullMasks.get());
            cudaDeviceSynchronize();
            CheckCudaError(cudaGetLastError());

            // Free old string buffers and replace with side buffers
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
    void ReconstructRawNumbers(std::vector<void*>& multiKeys,
                               std::vector<std::unique_ptr<int8_t[]>>& multiKeysNullMasks,
                               V* values,
                               int8_t* valuesNullMask,
                               int64_t* occurrences,
                               int32_t* outDataElementCount)
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

            // Copy key col pointer to CPU
            int8_t* keyNullSingleBuffer;
            GPUMemory::copyDeviceToHost(&keyNullSingleBuffer,
                                        reinterpret_cast<int8_t**>(keysNullBuffer_ + t), 1);

            GPUReconstruct::reconstructCol(multiKeysNullMasks[t].get(), outDataElementCount,
                                           keyNullSingleBuffer, occupancyMask.get(), maxHashCount_);
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
            GPUReconstruct::reconstructCol(valuesNullMask, outDataElementCount, valuesNullMask_,
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
    void GetResults(std::vector<void*>* outKeysVector,
                    O** outValues,
                    int32_t* outDataElementCount,
                    std::vector<int8_t*>* outKeysNullMasksVector = nullptr,
                    int8_t** outValuesNullMask = nullptr)
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
            if (outKeysNullMasksVector != nullptr)
            {
                // Copy key col pointer to CPU
                int8_t* keyNullSingleBuffer;
                GPUMemory::copyDeviceToHost(&keyNullSingleBuffer,
                                            reinterpret_cast<int8_t**>(keysNullBuffer_ + t), 1);

                // Reconstruct wide null mask
                int8_t* reconstructedNullMask;
                GPUReconstruct::reconstructColKeep(&reconstructedNullMask, outDataElementCount,
                                                   keyNullSingleBuffer, occupancyMask.get(), maxHashCount_);

                // Compress null mask
                int8_t* compressedNullMask;
                GPUMemory::allocAndSet(&compressedNullMask, 0,
                                       (*outDataElementCount + sizeof(int32_t) * 8 - 1) / (sizeof(int8_t) * 8));
                kernel_compress_null_mask<<<Context::getInstance().calcGridDim(*outDataElementCount),
                                            Context::getInstance().getBlockDim()>>>(
                    reinterpret_cast<int32_t*>(compressedNullMask), reconstructedNullMask, *outDataElementCount);
                GPUMemory::free(reconstructedNullMask);
                outKeysNullMasksVector->emplace_back(compressedNullMask);
            }
        }

        if (!std::is_same<AGG, AggregationFunctions::none>::value)
        {
            // Merge multipied arrays (values and occurrences)
            std::tuple<cuda_ptr<V>, cuda_ptr<int64_t>> mergedArrays =
                MergeMultipliedArrays<AGG, V, USE_VALUES, USE_KEY_OCCURRENCES>(values_, keyOccurrenceCount_,
                                                                               occupancyMask.get(), maxHashCount_,
                                                                               GB_ARRAY_MULTIPLIER);
            cuda_ptr<V> mergedValues = std::move(std::get<0>(mergedArrays));
            cuda_ptr<int64_t> mergedOccurrences = std::move(std::get<1>(mergedArrays));

            if (USE_VALUES)
            {
                cuda_ptr<int8_t> valuesNullMaskCompressed((maxHashCount_ + sizeof(int32_t) * 8 - 1) /
                                                              (sizeof(int8_t) * 8),
                                                          0);
                kernel_compress_null_mask<<<Context::getInstance().calcGridDim(maxHashCount_),
                                            Context::getInstance().getBlockDim()>>>(
                    reinterpret_cast<int32_t*>(valuesNullMaskCompressed.get()), valuesNullMask_, maxHashCount_);

                // Reconstruct aggregated values
                if (DIRECT_VALUES) // for min, max and sum: mergedValues.get() are direct results, just reconstruct them
                {
                    if (!std::is_same<O, V>::value)
                    {
                        CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                              "Input and output value data type must "
                                              "be the same in GROUP BY");
                    }
                    // reinterpret_cast is needed to solve compilation error
                    GPUReconstruct::reconstructColKeep(outValues, outDataElementCount,
                                                       reinterpret_cast<O*>(mergedValues.get()),
                                                       occupancyMask.get(), maxHashCount_, outValuesNullMask,
                                                       valuesNullMaskCompressed.get());
                }
                else if (std::is_same<AGG, AggregationFunctions::avg>::value) // for avg: mergedValues.get() need to be divided by keyOccurrences_ and reconstructed
                {
                    cuda_ptr<O> outValuesGPU(maxHashCount_);
                    // Divide by counts to get averages for buckets
                    try
                    {
                        GPUArithmetic::Arithmetic<ArithmeticOperations::div>(outValuesGPU.get(),
                                                                             mergedValues.get(),
                                                                             mergedOccurrences.get(),
                                                                             maxHashCount_);
                    }
                    catch (const query_engine_error& err)
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
                                                       occupancyMask.get(), maxHashCount_, outValuesNullMask,
                                                       valuesNullMaskCompressed.get());
                }
            }
            else if (std::is_same<AGG, AggregationFunctions::count>::value) // for count: reconstruct and return keyOccurrences_
            {
                if (!std::is_same<O, int64_t>::value)
                {
                    CheckQueryEngineError(GPU_EXTENSION_ERROR, "Output value data type in GROUP BY "
                                                               "with COUNT must be int64_t");
                }
                // reinterpret_cast is needed to solve compilation error
                // not reinterpreting anything here actually, outValues is int64_t** always in this else-branch
                GPUReconstruct::reconstructColKeep(reinterpret_cast<int64_t**>(outValues),
                                                   outDataElementCount, mergedOccurrences.get(),
                                                   occupancyMask.get(), maxHashCount_);
                if (outValuesNullMask)
                {
                    if (*outDataElementCount == 0)
                    {
                        *outValuesNullMask = nullptr;
                    }
                    else
                    {
                        GPUMemory::allocAndSet(outValuesNullMask, 0,
                                               (*outDataElementCount + sizeof(int8_t) * 8 - 1) /
                                                   (sizeof(int8_t) * 8));
                    }
                }
            }
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
                    std::vector<std::unique_ptr<IGroupBy>>& tables,
                    std::vector<int8_t*>* outKeysNullMasksVector = nullptr,
                    int8_t** outValuesNullMask = nullptr)
    {
        if (tables.size() <= 0) // invalid count of tables
        {
            CheckQueryEngineError(GPU_EXTENSION_ERROR, "Number of tables have to be at least 1.");
        }
        else if (tables.size() == 1 || tables[1].get() == nullptr) // just one table
        {
            GetResults(outKeysVector, outValues, outDataElementCount, outKeysNullMasksVector, outValuesNullMask);
        }
        else // more tables
        {
            int32_t oldDeviceId = Context::getInstance().getBoundDeviceID();

            std::vector<void*> multiKeysAllHost; // this vector is oriented orthogonally to others (vector of cols)
            std::vector<std::vector<int8_t>> keysNullMasksAllHost(keysColCount_); // this one too
            std::vector<V> valuesAllHost;
            std::vector<int8_t> valuesNullMaskAllHost;
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
                std::vector<std::unique_ptr<int8_t[]>> keysNullMasks;
                std::unique_ptr<V[]> values = std::make_unique<V[]>(table->GetMaxHashCount());
                std::unique_ptr<int8_t[]> valuesNullMask =
                    std::make_unique<int8_t[]>(table->GetMaxHashCount());
                std::unique_ptr<int64_t[]> occurrences =
                    std::make_unique<int64_t[]>(table->GetMaxHashCount());
                int32_t elementCount;
                for (int32_t t = 0; t < keysColCount_; t++)
                {
                    keysNullMasks.emplace_back(std::make_unique<int8_t[]>(table->GetMaxHashCount()));
                }

                Context::getInstance().bindDeviceToContext(i);
                // Reconstruct multi-keys, values and occurrences
                table->ReconstructRawNumbers(multiKeys, keysNullMasks, values.get(),
                                             valuesNullMask.get(), occurrences.get(), &elementCount);

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
                    keysNullMasksAllHost[t].insert(keysNullMasksAllHost[t].end(), keysNullMasks[t].get(),
                                                   keysNullMasks[t].get() + elementCount);
                }
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
                void** multiKeysAllGPU;
                int8_t** keysNullMasksAllGPU;
                std::vector<void*> hostPointersToKeysAll;
                std::vector<int8_t*> hostPointersToKeysNullMasksAll;
                AllocKeysBuffer(&multiKeysAllGPU, &keysNullMasksAllGPU, keyTypesHost, sumElementCount,
                                &hostPointersToKeysAll, &hostPointersToKeysNullMasksAll);
                cuda_ptr<V> valuesAllGPU(sumElementCount);
                cuda_ptr<int8_t> valuesNullMaskAllGPU(sumElementCount);
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
                    GPUMemory::copyHostToDevice(hostPointersToKeysNullMasksAll[t],
                                                keysNullMasksAllHost[t].data(), sumElementCount);
                }
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
                std::vector<cuda_ptr<int8_t>> compressedKeysNullMasksAllManaged;
                std::vector<int8_t*> compressedKeysNullMasksAllPtr;
                for (int32_t t = 0; t < keysColCount_; t++)
                {
                    cuda_ptr<int8_t> managed = std::move(
                        GPUReconstruct::CompressNullMask(hostPointersToKeysNullMasksAll[t], sumElementCount));
                    compressedKeysNullMasksAllPtr.emplace_back(managed.get());
                    compressedKeysNullMasksAllManaged.emplace_back(std::move(managed));
                }

                // Merge results
                if (DIRECT_VALUES) // for min, max and sum
                {
                    GPUGroupBy<AGG, O, std::vector<void*>, V> finalGroupBy(sumElementCount, keyTypesHost);
                    finalGroupBy.ProcessBlock(hostPointersToKeysAll, compressedKeysNullMasksAllPtr,
                                              valuesAllGPU.get(), sumElementCount,
                                              GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                                                  .get());
                    finalGroupBy.GetResults(outKeysVector, outValues, outDataElementCount,
                                            outKeysNullMasksVector, outValuesNullMask);
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
                    sumGroupBy.ProcessBlock(hostPointersToKeysAll, compressedKeysNullMasksAllPtr,
                                            valuesAllGPU.get(), sumElementCount,
                                            GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                                                .get());
                    sumGroupBy.GetResults(&keysToDiscard, &valuesMerged, outDataElementCount);
                    FreeKeysVector(keysToDiscard, keyTypesHost);

                    // Calculate sum of occurences
                    // Initialize countGroupBy table with already existing keys from sumGroupBy - to guarantee the same order
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::vector<void*>, int64_t> countGroupBy(
                        sumElementCount, keyTypesHost, sumGroupBy.sourceIndices_,
                        sumGroupBy.keysBuffer_, sumGroupBy.keysNullBuffer_);
                    countGroupBy.ProcessBlock(hostPointersToKeysAll, compressedKeysNullMasksAllPtr,
                                              occurrencesAllGPU.get(), sumElementCount,
                                              GPUReconstruct::CompressNullMask(valuesNullMaskAllGPU.get(), sumElementCount)
                                                  .get());
                    countGroupBy.GetResults(outKeysVector, &occurencesMerged, outDataElementCount,
                                            outKeysNullMasksVector, outValuesNullMask);

                    // Divide merged values by merged occurences to get final averages
                    GPUMemory::alloc(outValues, *outDataElementCount);
                    try
                    {
                        GPUArithmetic::Arithmetic<ArithmeticOperations::div>(*outValues, valuesMerged, occurencesMerged,
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
                    GPUMemory::free(occurencesMerged);
                }
                else if (std::is_same<AGG, AggregationFunctions::count>::value) // for count
                {
                    if (!std::is_same<O, int64_t>::value)
                    {
                        CheckQueryEngineError(GPU_EXTENSION_ERROR,
                                              "Output value data type in GROUP BY with COUNT must "
                                              "be int64_t");
                    }
                    GPUGroupBy<AggregationFunctions::sum, int64_t, std::vector<void*>, int64_t> finalGroupBy(
                        sumElementCount, keyTypesHost);
                    finalGroupBy.ProcessBlock(hostPointersToKeysAll, compressedKeysNullMasksAllPtr,
                                              occurrencesAllGPU.get(), sumElementCount, nullptr);
                    // reinterpret_cast is needed to solve compilation error
                    finalGroupBy.GetResults(outKeysVector, reinterpret_cast<int64_t**>(outValues),
                                            outDataElementCount, outKeysNullMasksVector, outValuesNullMask);
                }

                else // for group by withou aggregation function
                {
                    GPUGroupBy<AGG, O, std::vector<void*>, V> finalGroupBy(sumElementCount, keyTypesHost);
                    finalGroupBy.ProcessBlock(hostPointersToKeysAll, compressedKeysNullMasksAllPtr,
                                              nullptr, sumElementCount, nullptr);
                    finalGroupBy.GetResults(outKeysVector, outValues, outDataElementCount,
                                            outKeysNullMasksVector, outValuesNullMask);
                }

                FreeKeysBuffer(multiKeysAllGPU, keysNullMasksAllGPU, keyTypes_, keysColCount_);
            }
            else
            {
                *outDataElementCount = 0;
            }
        }
    }
};
