#include "StringFactory.h"

GPUMemory::GPUString StringFactory::PrepareGPUString(const std::vector<std::string>& strings)
{
    std::vector<int64_t> stringIndices;
    std::string concat;

    int64_t prefixSum = 0;

    for (auto& str : strings)
    {
        prefixSum += str.size();
        stringIndices.push_back(prefixSum);
        concat += str;
    }

    GPUMemory::GPUString gpuString;
    GPUMemory::alloc(&gpuString.stringIndices, stringIndices.size());
    GPUMemory::copyHostToDevice(gpuString.stringIndices, stringIndices.data(), stringIndices.size());

    GPUMemory::alloc(&gpuString.allChars, prefixSum);
    GPUMemory::copyHostToDevice(gpuString.allChars, concat.data(), prefixSum);

    return gpuString;
}

GPUMemory::GPUString StringFactory::PrepareGPUString(const std::vector<std::string>& strings,
                                                     const std::string& databaseName,
                                                     const std::string& columnName,
                                                     size_t blockIndex,
                                                     int64_t loadSize,
                                                     int64_t loadOffset)
{
    std::vector<int64_t> stringIndices;
    std::string concat;

    int64_t prefixSum = 0;

    for (auto& str : strings)
    {
        prefixSum += str.size();
        stringIndices.push_back(prefixSum);
        concat += str;
    }

    GPUMemory::GPUString gpuString;

    gpuString.stringIndices = std::get<0>(Context::getInstance().getCacheForCurrentDevice().getColumn<int64_t>(
        databaseName, columnName + "_stringIndices", blockIndex, stringIndices.size(), loadSize, loadOffset));
    GPUMemory::copyHostToDevice(gpuString.stringIndices, stringIndices.data(), stringIndices.size());

    gpuString.allChars = std::get<0>(Context::getInstance().getCacheForCurrentDevice().getColumn<char>(
        databaseName, columnName + "_allChars", blockIndex, concat.size(), loadSize, loadOffset));
    GPUMemory::copyHostToDevice(gpuString.allChars, concat.data(), prefixSum);

    return gpuString;
}
