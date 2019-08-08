#include "GPUGroupByMultiKey.cuh"

__device__ int32_t GetHash(DataType* keyTypes, int32_t keysColCount, void** inKeys, int32_t i, const int32_t hashCoef)
{
    uint32_t crc = 0xFFFFFFFF;

    for (int32_t t = 0; t < keysColCount; t++)
    {
        uint32_t hash = 0;
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
            hash = reinterpret_cast<uint32_t*>(inKeys[t])[i];
            break;
        case DataType::COLUMN_LONG:
            hash = static_cast<uint32_t>(reinterpret_cast<int64_t*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_FLOAT:
            hash = static_cast<uint32_t>(reinterpret_cast<float*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_DOUBLE:
            hash = static_cast<uint32_t>(reinterpret_cast<double*>(inKeys[t])[i]);
            break;
        case DataType::COLUMN_STRING:
        {
            GPUMemory::GPUString strCol = *reinterpret_cast<GPUMemory::GPUString*>(inKeys[t]);
            hash = GetHash(strCol.allChars + GetStringIndex(strCol.stringIndices, i),
                           GetStringLength(strCol.stringIndices, i));
        }
        break;
        case DataType::COLUMN_INT8_T:
            hash = static_cast<uint32_t>(reinterpret_cast<int8_t*>(inKeys[t])[i]);
            break;
        default:
            hash = 0;
            break;
        }
        crc = (CRC_32_TAB[((crc >> 24) ^ hash) & 0xFF] ^ (crc << 8));
    }
    return (crc >> 16) ^ (crc & 0xFFFF);
}


__device__ bool
AreEqualMultiKeys(DataType* keyTypes, int32_t keysColCount, void** keysA, int32_t indexA, void** keysB, int32_t indexB)
{
    for (int32_t t = 0; t < keysColCount; t++)
    {
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
            if (reinterpret_cast<int32_t*>(keysA[t])[indexA] != reinterpret_cast<int32_t*>(keysB[t])[indexB])
            {
                return false;
            }
            break;
        case DataType::COLUMN_LONG:
            if (reinterpret_cast<int64_t*>(keysA[t])[indexA] != reinterpret_cast<int64_t*>(keysB[t])[indexB])
            {
                return false;
            }
            break;
        case DataType::COLUMN_FLOAT:
            if (reinterpret_cast<float*>(keysA[t])[indexA] != reinterpret_cast<float*>(keysB[t])[indexB])
            {
                return false;
            }
            break;
        case DataType::COLUMN_DOUBLE:
            if (reinterpret_cast<double*>(keysA[t])[indexA] != reinterpret_cast<double*>(keysB[t])[indexB])
            {
                return false;
            }
            break;
        case DataType::COLUMN_STRING:
        {
            GPUMemory::GPUString strColA = *reinterpret_cast<GPUMemory::GPUString*>(keysA[t]);
            GPUMemory::GPUString strColB = *reinterpret_cast<GPUMemory::GPUString*>(keysB[t]);
            if (!AreEqualStrings(strColA.allChars + GetStringIndex(strColA.stringIndices, indexA),
                                 GetStringLength(strColA.stringIndices, indexA), strColB, indexB))
            {
                return false;
            }
            break;
        }
        case DataType::COLUMN_INT8_T:
            if (reinterpret_cast<int8_t*>(keysA[t])[indexA] != reinterpret_cast<int8_t*>(keysB[t])[indexB])
            {
                return false;
            }
            break;
        default:
            break;
        }
    }
    return true;
}


__device__ bool
IsNewMultiKey(DataType* keyTypes, int32_t keysColCount, void** inKeys, int32_t i, void** keysBuffer, int32_t* sourceIndices, int32_t index)
{
    return (sourceIndices[index] >= 0 &&
            !AreEqualMultiKeys(keyTypes, keysColCount, inKeys, i, inKeys, sourceIndices[index])) ||
           (sourceIndices[index] == GBS_SOURCE_INDEX_KEY_IN_BUFFER &&
            !AreEqualMultiKeys(keyTypes, keysColCount, inKeys, i, keysBuffer, index));
}


template <>
void ReconstructSingleKeyColKeep<std::string>(std::vector<void*>* outKeysVector,
                                              int32_t* outDataElementCount,
                                              int8_t* occupancyMaskPtr,
                                              void** keyCol,
                                              int32_t elementCount)
{
    // Copy struct (we need to get pointer to struct at first)
    GPUMemory::GPUString* structPointer;
    GPUMemory::copyDeviceToHost(&structPointer, reinterpret_cast<GPUMemory::GPUString**>(keyCol), 1);
    GPUMemory::GPUString keyBufferSingleCol;
    GPUMemory::copyDeviceToHost(&keyBufferSingleCol, structPointer, 1);

    // Reconstruct string keys
    GPUMemory::GPUString* outKeysSingleCol = new GPUMemory::GPUString();
    GPUReconstruct::ReconstructStringColKeep(outKeysSingleCol, outDataElementCount,
                                             keyBufferSingleCol, occupancyMaskPtr, elementCount);

    outKeysVector->emplace_back(outKeysSingleCol);
}


template <>
void ReconstructSingleKeyCol<std::string>(std::vector<void*>* outKeysVector,
                                          int32_t* outDataElementCount,
                                          int8_t* occupancyMaskPtr,
                                          void** keyCol,
                                          int32_t elementCount)
{
    // Copy struct (we need to get pointer to struct at first)
    GPUMemory::GPUString* structPointer;
    GPUMemory::copyDeviceToHost(&structPointer, reinterpret_cast<GPUMemory::GPUString**>(keyCol), 1);
    GPUMemory::GPUString keyBufferSingleCol;
    GPUMemory::copyDeviceToHost(&keyBufferSingleCol, structPointer, 1);

    // Reconstruct string as raw
    std::vector<int32_t> stringLengths;
    std::vector<char> allChars;
    GPUReconstruct::ReconstructStringColRaw(stringLengths, allChars, outDataElementCount,
                                            keyBufferSingleCol, occupancyMaskPtr, elementCount);

    CPUString* outKeysSingleCol = new CPUString{stringLengths, allChars};
    outKeysVector->emplace_back(outKeysSingleCol);
}


void AllocKeysBuffer(void*** keysBuffer, std::vector<DataType> keyTypes, int32_t rowCount, std::vector<void*>* pointers)
{
    GPUMemory::alloc(keysBuffer, keyTypes.size());
    for (int32_t i = 0; i < keyTypes.size(); i++)
    {
        switch (keyTypes[i])
        {
        case DataType::COLUMN_INT:
        {
            int32_t* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, rowCount);
            GPUMemory::copyHostToDevice(reinterpret_cast<int32_t**>(*keysBuffer + i), &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        case DataType::COLUMN_LONG:
        {
            int64_t* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, rowCount);
            GPUMemory::copyHostToDevice(reinterpret_cast<int64_t**>(*keysBuffer + i), &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        case DataType::COLUMN_FLOAT:
        {
            float* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, rowCount);
            GPUMemory::copyHostToDevice(reinterpret_cast<float**>(*keysBuffer + i), &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        case DataType::COLUMN_DOUBLE:
        {
            double* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, rowCount);
            GPUMemory::copyHostToDevice(reinterpret_cast<double**>(*keysBuffer + i), &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        case DataType::COLUMN_STRING:
        {
            GPUMemory::GPUString emptyStringCol{nullptr, nullptr};
            GPUMemory::GPUString* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, 1);
            GPUMemory::copyHostToDevice(gpuKeyCol, &emptyStringCol, 1);
            GPUMemory::copyHostToDevice(reinterpret_cast<GPUMemory::GPUString**>(*keysBuffer + i),
                                        &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        case DataType::COLUMN_INT8_T:
        {
            int8_t* gpuKeyCol;
            GPUMemory::alloc(&gpuKeyCol, rowCount);
            GPUMemory::copyHostToDevice(reinterpret_cast<int8_t**>(*keysBuffer + i), &gpuKeyCol, 1);
            if (pointers)
            {
                pointers->emplace_back(gpuKeyCol);
            }
            break;
        }
        default:
            CheckQueryEngineError(GPU_EXTENSION_ERROR, "Multi-key GROUP BY with keys of type " +
                                                           std::to_string(keyTypes[i]) + " is not supported");
            break;
        }
    }
}

void FreeKeysBuffer(void** keysBuffer, DataType* keyTypes, int32_t keysColCount)
{
    // Copy data types back from GPU
    std::vector<DataType> keyTypesHost;
    keyTypesHost.resize(keysColCount);
    GPUMemory::copyDeviceToHost(keyTypesHost.data(), keyTypes, keysColCount);

    for (int32_t i = 0; i < keysColCount; i++)
    {
        void* ptr;
        GPUMemory::copyDeviceToHost(&ptr, keysBuffer + i, 1); // copy single pointer
        if (ptr)
        {
            if (keyTypesHost[i] == DataType::COLUMN_STRING)
            {
                GPUMemory::GPUString str;
                GPUMemory::copyDeviceToHost(&str, reinterpret_cast<GPUMemory::GPUString*>(ptr), 1);
                GPUMemory::free(str);
            }
            GPUMemory::free(ptr);
        }
    }
    GPUMemory::free(keysBuffer);
}

void FreeKeysVector(std::vector<void*> keysVector, std::vector<DataType> keyTypes)
{
    for (int32_t i = 0; i < keyTypes.size(); i++)
    {
        if (keysVector[i])
        {
            if (keyTypes[i] == DataType::COLUMN_STRING)
            {
                GPUMemory::GPUString* str = reinterpret_cast<GPUMemory::GPUString*>(keysVector[i]);
                GPUMemory::free(*str);
                delete str;
            }
            GPUMemory::free(keysVector[i]);
        }
    }
}


__global__ void kernel_collect_string_lengths(int32_t* stringLengths,
                                              int32_t* sourceIndices,
                                              GPUMemory::GPUString** inKeysSingleCol,
                                              GPUMemory::GPUString** keysBufferSingleCol,
                                              int32_t maxHashCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maxHashCount; i += stride)
    {
        if (sourceIndices[i] >= 0) // string from input key array
        {
            stringLengths[i] = GetStringLength((*inKeysSingleCol)->stringIndices, sourceIndices[i]);
        }
        else if (sourceIndices[i] == GBS_SOURCE_INDEX_KEY_IN_BUFFER) // string stored in key buffer
        {
            stringLengths[i] = GetStringLength((*keysBufferSingleCol)->stringIndices, i);
        }
        else // GBS_SOURCE_INDEX_EMPTY_KEY - no string
        {
            stringLengths[i] = 0;
        }
    }
}


__global__ void kernel_collect_multi_keys(DataType* keyTypes,
                                          int32_t keysColCount,
                                          int32_t* sourceIndices,
                                          void** keysBuffer,
                                          GPUMemory::GPUString* stringSideBuffers,
                                          int32_t** stringLengthsBuffers,
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
                    reinterpret_cast<int32_t*>(keysBuffer[t])[i] =
                        reinterpret_cast<int32_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_LONG:
                    reinterpret_cast<int64_t*>(keysBuffer[t])[i] =
                        reinterpret_cast<int64_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_FLOAT:
                    reinterpret_cast<float*>(keysBuffer[t])[i] =
                        reinterpret_cast<float*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_DOUBLE:
                    reinterpret_cast<double*>(keysBuffer[t])[i] =
                        reinterpret_cast<double*>(inKeys[t])[sourceIndices[i]];
                    break;
                case DataType::COLUMN_STRING:
                {
                    // Copy strings from inKeys according to sourceIndices
                    GPUMemory::GPUString& sideBufferStr = stringSideBuffers[t];
                    GPUMemory::GPUString& inKeysStr = *reinterpret_cast<GPUMemory::GPUString*>(inKeys[t]);
                    for (int32_t j = 0; j < stringLengthsBuffers[t][i]; j++)
                    {
                        sideBufferStr.allChars[GetStringIndex(sideBufferStr.stringIndices, i) + j] =
                            inKeysStr.allChars[GetStringIndex(inKeysStr.stringIndices, sourceIndices[i]) + j];
                    }
                    break;
                }
                case DataType::COLUMN_INT8_T:
                    reinterpret_cast<int8_t*>(keysBuffer[t])[i] =
                        reinterpret_cast<int8_t*>(inKeys[t])[sourceIndices[i]];
                    break;
                default:
                    break;
                }
            }
            sourceIndices[i] = GBS_SOURCE_INDEX_KEY_IN_BUFFER; // Mark as stored in keyBuffer
        }
        else if (sourceIndices[i] == GBS_SOURCE_INDEX_KEY_IN_BUFFER)
        {
            for (int32_t t = 0; t < keysColCount; t++)
            {
                if (keyTypes[t] == DataType::COLUMN_STRING)
                {
                    // Copy strings from keysBuffer
                    GPUMemory::GPUString& sideBufferStr = stringSideBuffers[t];
                    GPUMemory::GPUString& keysBufferStr =
                        *reinterpret_cast<GPUMemory::GPUString*>(keysBuffer[t]);
                    for (int32_t j = 0; j < stringLengthsBuffers[t][i]; j++)
                    {
                        sideBufferStr.allChars[GetStringIndex(sideBufferStr.stringIndices, i) + j] =
                            keysBufferStr.allChars[GetStringIndex(keysBufferStr.stringIndices, i) + j];
                    }
                }
            }
        }
    }
}
