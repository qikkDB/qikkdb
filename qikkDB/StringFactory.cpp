#include "StringFactory.h"

GPUMemory::GPUString StringFactory::PrepareGPUString(const std::string* strings, const size_t stringCount)
{
    std::vector<int64_t> stringIndices;
    std::string concat;

    int64_t prefixSum = 0;

    for (size_t i = 0; i < stringCount; i++)
    {
        prefixSum += strings[i].size();
        stringIndices.push_back(prefixSum);
        concat += strings[i];
    }

    GPUMemory::GPUString gpuString;
    GPUMemory::alloc(&gpuString.stringIndices, stringIndices.size());
    GPUMemory::copyHostToDevice(gpuString.stringIndices, stringIndices.data(), stringIndices.size());

    if (prefixSum == 0)
    {
        // Don't allocate empty buffer
        gpuString.allChars = nullptr;
    }
    else
    {
        GPUMemory::alloc(&gpuString.allChars, prefixSum);
        GPUMemory::copyHostToDevice(gpuString.allChars, concat.data(), prefixSum);
    }

    return gpuString;
}

GPUMemory::GPUString StringFactory::PrepareGPUString(const std::string* strings,
                                                     const size_t stringCount,
                                                     const std::string& databaseName,
                                                     const std::string& columnName,
                                                     size_t blockIndex,
                                                     int64_t loadSize,
                                                     int64_t loadOffset)
{
    std::vector<int64_t> stringIndices;
    std::string concat;

    int64_t prefixSum = 0;

    for (size_t i = 0; i < stringCount; i++)
    {
        prefixSum += strings[i].size();
        stringIndices.push_back(prefixSum);
        concat += strings[i];
    }

    GPUMemory::GPUString gpuString;

    // If no chars are in the concat, don't use cache
    if (prefixSum == 0)
    {
        GPUMemory::alloc(&gpuString.stringIndices, stringIndices.size());
        GPUMemory::copyHostToDevice(gpuString.stringIndices, stringIndices.data(), stringIndices.size());
        // Don't allocate empty buffer
        gpuString.allChars = nullptr;
    }
    else
    {
        gpuString.stringIndices = std::get<0>(Context::getInstance().getCacheForCurrentDevice().GetColumn<int64_t>(
            databaseName, columnName + "_stringIndices", blockIndex, stringIndices.size(), loadSize, loadOffset));
        GPUMemory::copyHostToDevice(gpuString.stringIndices, stringIndices.data(), stringIndices.size());

        gpuString.allChars = std::get<0>(Context::getInstance().getCacheForCurrentDevice().GetColumn<char>(
            databaseName, columnName + "_allChars", blockIndex, concat.size(), loadSize, loadOffset));
        GPUMemory::copyHostToDevice(gpuString.allChars, concat.data(), prefixSum);
    }

    return gpuString;
}
