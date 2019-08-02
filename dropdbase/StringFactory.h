#pragma once

#include "QueryEngine/GPUCore/GPUMemory.cuh"

class StringFactory
{
public:
    static GPUMemory::GPUString PrepareGPUString(const std::vector<std::string>& strings);
    static GPUMemory::GPUString PrepareGPUString(const std::vector<std::string>& strings,
                                                 const std::string& databaseName,
                                                 const std::string& columnName,
                                                 size_t blockIndex);
};