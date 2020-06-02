#include "GPUGroupByString.cuh"


__device__ int32_t GetHash(char* text, int32_t length)
{
    int32_t hash = 0;
    for (int32_t i = 0; i < length; i++)
    {
        hash = GBS_STRING_HASH_COEF * hash + text[i];
    }
    return hash;
}


__device__ bool AreEqualStrings(char* textA, int32_t lenghtA, GPUMemory::GPUString stringColB, int64_t indexB)
{
    return FilterConditions::equal{}.compareStrings(textA, lenghtA,
                                                    stringColB.allChars +
                                                        GetStringIndex(stringColB.stringIndices, indexB),
                                                    GetStringLength(stringColB.stringIndices, indexB));
}


__device__ bool IsNewKey(char* checkedKeyChars,
                         int32_t checkedKeyLength,
                         GPUMemory::GPUString inKeys,
                         GPUMemory::GPUString keysBuffer,
                         int32_t* sourceIndices,
                         int32_t index)
{
    return (sourceIndices[index] >= 0 &&
            !AreEqualStrings(checkedKeyChars, checkedKeyLength, inKeys, sourceIndices[index])) || // find in inKeys at first
           (sourceIndices[index] == GBS_SOURCE_INDEX_KEY_IN_BUFFER &&
            !AreEqualStrings(checkedKeyChars, checkedKeyLength, keysBuffer, index)); // and in keysBuffer next
}


__global__ void kernel_collect_string_keys(GPUMemory::GPUString sideBuffer,
                                           int32_t* sourceIndices,
                                           int32_t* stringLengths,
                                           GPUMemory::GPUString keysBuffer,
                                           int32_t maxHashCount,
                                           GPUMemory::GPUString inKeys,
                                           int32_t inKeysCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maxHashCount; i += stride)
    {
        if (sourceIndices[i] >= 0)
        {
            for (int32_t j = 0; j < stringLengths[i]; j++)
            {
                sideBuffer.allChars[GetStringIndex(sideBuffer.stringIndices, i) + j] =
                    inKeys.allChars[GetStringIndex(inKeys.stringIndices, sourceIndices[i]) + j];
            }
        }
        else if (sourceIndices[i] == GBS_SOURCE_INDEX_KEY_IN_BUFFER)
        {
            for (int32_t j = 0; j < stringLengths[i]; j++)
            {
                sideBuffer.allChars[GetStringIndex(sideBuffer.stringIndices, i) + j] =
                    keysBuffer.allChars[GetStringIndex(keysBuffer.stringIndices, i) + j];
            }
        }
    }
}


__global__ void kernel_source_indices_to_mask(int8_t* occupancyMask, int32_t* sourceIndices, int32_t maxHashCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maxHashCount; i += stride)
    {
        occupancyMask[i] = (sourceIndices[i] != GBS_SOURCE_INDEX_EMPTY_KEY);
    }
}


__global__ void kernel_mark_collected_strings(int32_t* sourceIndices, int32_t maxHashCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < maxHashCount; i += stride)
    {
        if (sourceIndices[i] >= 0)
        {
            sourceIndices[i] = GBS_SOURCE_INDEX_KEY_IN_BUFFER;
        }
    }
}
