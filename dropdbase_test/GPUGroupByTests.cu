#include <cstdint>
#include <cstdlib>
#include <memory>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUGroupByString.cuh"
#include "../dropdbase/QueryEngine/GPUCore/AggregationFunctions.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"
#include "../dropdbase/StringFactory.h"
#include "gtest/gtest.h"

template <typename AGG>
void TestGroupByString(std::vector<std::vector<std::string>> keys,
                       std::vector<std::vector<int32_t>> values,
                       std::unordered_map<std::string, int32_t> correctPairs)
{
    constexpr int32_t hashTableSize = 8;
    GPUGroupBy<AGG, int32_t, std::string, int32_t> groupBy(hashTableSize);
    for (int32_t b = 0; b < keys.size(); b++) // per "block"
    {
        // std::cout << "BLOCK " << b << ":" << std::endl;
        int32_t dataElementCount = min(keys[b].size(), values[b].size());
        GPUMemory::GPUString gpuInKeys = StringFactory::PrepareGPUString(keys[b]);
        cuda_ptr<int32_t> gpuInValues(dataElementCount);
        GPUMemory::copyHostToDevice(gpuInValues.get(), values[b].data(), dataElementCount);

        groupBy.groupBy(gpuInKeys, gpuInValues.get(), dataElementCount);
        GPUMemory::free(gpuInKeys);
        /*
        // DEBUG prints
        int32_t sourceIds[hashTableSize];
        int32_t stringLens[hashTableSize];
        GPUMemory::copyDeviceToHost(sourceIds, groupBy.sourceIndices_, hashTableSize);
        GPUMemory::copyDeviceToHost(stringLens, groupBy.stringLengths_, hashTableSize);
        for (int32_t i = 0; i < hashTableSize; i++)
        {
            std::cout << stringLens[i] << "  " << sourceIds[i];
            if (sourceIds[i] >= 0)
            {
                std::cout << " (" << keys[b][sourceIds[i]] << ")";
            }
            std::cout << std::endl;
        }
        */
    }
    GPUMemory::GPUString resultKeysGpu;
    int32_t* resultValuesGpu;
    int32_t resultCount;
    groupBy.getResults(&resultKeysGpu, &resultValuesGpu, &resultCount);
    std::unique_ptr<std::string[]> resultKeys = std::make_unique<std::string[]>(hashTableSize);
    std::unique_ptr<int32_t[]> resultValues = std::make_unique<int32_t[]>(hashTableSize);
    GPUReconstruct::ReconstructStringCol(resultKeys.get(), &resultCount, resultKeysGpu, nullptr, resultCount);
    GPUMemory::copyDeviceToHost(resultValues.get(), resultValuesGpu, resultCount);

    ASSERT_EQ(correctPairs.size(), resultCount) << " wrong number of keys";
    for (int32_t i = 0; i < resultCount; i++)
    {
        ASSERT_FALSE(correctPairs.find(resultKeys[i]) == correctPairs.end())
            << " key \"" << resultKeys[i] << "\"";
        ASSERT_EQ(correctPairs[resultKeys[i]], resultValues[i]) << " at key \"" << resultKeys[i] << "\"";
    }
    GPUMemory::free(resultKeysGpu);
    GPUMemory::free(resultValuesGpu);
}


TEST(GPUGroupByTests, StringUnique)
{
    TestGroupByString<AggregationFunctions::sum>({{"Apple", "Abcd", "XYZ"}}, {{1, 2, -1}},
                                                 {{"Apple", 1}, {"Abcd", 2}, {"XYZ", -1}});
}

TEST(GPUGroupByTests, StringSimple)
{
    TestGroupByString<AggregationFunctions::sum>({{"Apple", "Abcd", "XYZ", "Abcd", "ZYX", "XYZ"}},
                                                 {{1, 2, -1, 3, 7, 5}},
                                                 {{"Apple", 1}, {"Abcd", 5}, {"XYZ", 4}, {"ZYX", 7}});
}

TEST(GPUGroupByTests, StringMultiBlockSimple)
{
    TestGroupByString<AggregationFunctions::sum>({{"Apple", "Abcd"}, {"XYZ", "Abcd"}, {"ZYX", "XYZ"}, {"Apple", "Apple"}},
                                                 {{1, 1}, {1, 2}, {1, 2}, {2, 4}},
                                                 {{"Apple", 7}, {"Abcd", 3}, {"XYZ", 3}, {"ZYX", 1}});
}

TEST(GPUGroupByTests, StringMultiBlockMediumSum)
{
    TestGroupByString<AggregationFunctions::sum>(
        {{"Apple", "Abcd", "Apple", "XYZ"}, {"Banana", "XYZ", "Abcd", "0"}, {"XYZ", "XYZ"}},
        {{1, 2, 3, 4}, {5, 6, 7, 10}, {13, 15}},
        {{"Apple", 4}, {"Abcd", 9}, {"Banana", 5}, {"XYZ", 38}, {"0", 10}});
}

TEST(GPUGroupByTests, StringMultiBlockMediumMin)
{
    TestGroupByString<AggregationFunctions::min>(
        {{"Apple", "Abcd", "Apple", "XYZ"}, {"Banana", "XYZ", "Abcd", "0"}, {"XYZ", "XYZ"}},
        {{1, 2, 3, 4}, {5, 6, 7, 10}, {13, 15}},
        {{"Apple", 1}, {"Abcd", 2}, {"Banana", 5}, {"XYZ", 4}, {"0", 10}});
}

TEST(GPUGroupByTests, StringMultiBlockMediumMax)
{
    TestGroupByString<AggregationFunctions::max>(
        {{"Apple", "Abcd", "Apple", "XYZ"}, {"Banana", "XYZ", "Abcd", "0"}, {"XYZ", "XYZ"}},
        {{1, 2, 3, 4}, {5, 6, 7, 10}, {13, 15}},
        {{"Apple", 3}, {"Abcd", 7}, {"Banana", 5}, {"XYZ", 15}, {"0", 10}});
}
