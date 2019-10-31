#pragma once

#include "QueryEngine/GPUCore/GPUMemory.cuh"

class StringFactory
{
public:
    static GPUMemory::GPUString PrepareGPUString(const std::string* strings, size_t stringCount);
    static GPUMemory::GPUString PrepareGPUString(const std::string* strings,
                                                 size_t stringCount,
                                                 const std::string& databaseName,
                                                 const std::string& columnName,
                                                 size_t blockIndex,
                                                 int64_t loadSize,
                                                 int64_t loadOffset);
};