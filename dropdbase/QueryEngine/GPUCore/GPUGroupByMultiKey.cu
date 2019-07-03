#include "GPUGroupByMultiKey.cuh"


__device__ int32_t GetHash(DataType* keyTypes, int32_t keysColCount, void** inKeys, int32_t i)
{
    int32_t hash = 0;
    for (int32_t t = 0; t < keysColCount; t++)
    {
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
            hash ^= reinterpret_cast<int32_t*>(inKeys[t])[i];
            break;
        case DataType::COLUMN_LONG:
            hash ^= static_cast<int32_t>(reinterpret_cast<int64_t*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_FLOAT:
            hash ^= static_cast<int32_t>(reinterpret_cast<float*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_DOUBLE:
            hash ^= static_cast<int32_t>(reinterpret_cast<double*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_STRING:
            // TODO implement
            break;
        case DataType::COLUMN_INT8_T:
            hash ^= static_cast<int32_t>(reinterpret_cast<int8_t*>(inKeys[t])[i]);
            break;
        default:
            break;
        }
    }
    return hash;
}


__device__ bool AreEqualMultiKeys(DataType* keyTypes,
    int32_t keysColCount,
    void** keysA,
    int32_t indexA,
    void** keysB,
    int32_t indexB)
{
    bool equals = true;
    for (int32_t t = 0; t < keysColCount; t++)
    {
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
            equals &= (reinterpret_cast<int32_t*>(keysA[t])[indexA] == reinterpret_cast<int32_t*>(keysB[t])[indexB]);
            break;
        case DataType::COLUMN_LONG:
            equals &= (reinterpret_cast<int64_t*>(keysA[t])[indexA] == reinterpret_cast<int64_t*>(keysB[t])[indexB]);
            break;
        case DataType::COLUMN_FLOAT:
            equals &= (reinterpret_cast<float*>(keysA[t])[indexA] == reinterpret_cast<float*>(keysB[t])[indexB]);
            break;
        case DataType::COLUMN_DOUBLE:
            equals &= (reinterpret_cast<double*>(keysA[t])[indexA] == reinterpret_cast<double*>(keysB[t])[indexB]);
            break;
        case DataType::COLUMN_STRING:
            // TODO implement
            break;
        case DataType::COLUMN_INT8_T:
            equals &= (reinterpret_cast<int8_t*>(keysA[t])[indexA] == reinterpret_cast<int8_t*>(keysB[t])[indexB]);
            break;
        default:
            break;
        }
    }
    return equals;
}


__device__ bool IsNewMultiKey(DataType* keyTypes,
    int32_t keysColCount,
    void** inKeys,
    int32_t i,
    void** keysBuffer,
    int32_t* sourceIndices,
    int32_t index)
{
    return (sourceIndices[index] >= 0 &&
        !AreEqualMultiKeys(keyTypes, keysColCount, inKeys, i, inKeys, sourceIndices[index])) ||
        (sourceIndices[index] == GBS_SOURCE_INDEX_KEY_IN_BUFFER &&
        !AreEqualMultiKeys(keyTypes, keysColCount, inKeys, i, keysBuffer, index));
}


__global__ void kernel_collect_multi_keys(DataType* keyTypes,
    int32_t keysColCount,
    int32_t* sourceIndices,
    void** keysBuffer,
    int32_t maxHashCount,
    void** inKeys)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maxHashCount; i += stride)
    {
        if (sourceIndices[i] >= 0)
        {
            for (int32_t t = 0; t < keysColCount; t++)
            {
                switch (keyTypes[t])
                {
                case DataType::COLUMN_INT:
                    reinterpret_cast<int32_t*>(keysBuffer[t])[i] = reinterpret_cast<int32_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_LONG:
                    reinterpret_cast<int64_t*>(keysBuffer[t])[i] = reinterpret_cast<int64_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_FLOAT:
                    reinterpret_cast<float*>(keysBuffer[t])[i] = reinterpret_cast<float*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_DOUBLE:
                    reinterpret_cast<double*>(keysBuffer[t])[i] = reinterpret_cast<double*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_STRING:
                    // TODO implement
                    break;
                case DataType::COLUMN_INT8_T:
                    reinterpret_cast<int8_t*>(keysBuffer[t])[i] = reinterpret_cast<int8_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                default:
                    break;
                }
            }
            sourceIndices[i] = GBS_SOURCE_INDEX_KEY_IN_BUFFER; // Mark as stored in keyBuffer
        }
    }
}
