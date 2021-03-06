#include <cstdint>
#include <cstdlib>
#include <memory>

#include "../qikkDB/QueryEngine/Context.h"
#include "../qikkDB/QueryEngine/GPUCore/GPUGroupByString.cuh"
#include "../qikkDB/QueryEngine/GPUCore/GPUGroupByMultiKey.cuh"
#include "../qikkDB/QueryEngine/GPUCore/AggregationFunctions.cuh"
#include "../qikkDB/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../qikkDB/QueryEngine/GPUCore/cuda_ptr.h"
#include "../qikkDB/DataType.h"
#include "../qikkDB/StringFactory.h"
#include "gtest/gtest.h"

template <typename AGG>
void TestGroupByString(std::vector<std::vector<std::string>> keys,
                       std::vector<std::vector<int32_t>> values,
                       std::unordered_map<std::string, int32_t> correctPairs)
{
    constexpr int32_t hashTableSize = 8;
    GPUGroupBy<AGG, int32_t, std::string, int32_t> groupBy(hashTableSize, 1);
    for (int32_t b = 0; b < keys.size(); b++) // per "block"
    {
        // std::cout << "BLOCK " << b << ":" << std::endl;
        int32_t dataElementCount = min(keys[b].size(), values[b].size());
        GPUMemory::GPUString gpuInKeys = StringFactory::PrepareGPUString(keys[b].data(), keys[b].size());
        cuda_ptr<int32_t> gpuInValues(dataElementCount);
        GPUMemory::copyHostToDevice(gpuInValues.get(), values[b].data(), dataElementCount);

        groupBy.ProcessBlock(gpuInKeys, gpuInValues.get(), dataElementCount);
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
    groupBy.GetResults(&resultKeysGpu, &resultValuesGpu, &resultCount);
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


/// Print all single keys of multiple-key on specified index
void DebugPrintMultiKey(std::vector<DataType> keyTypes, std::vector<void*> keys, int32_t keyIndex, std::string lineEnd = "\n")
{
    std::cout << "key: {";
    for (int32_t t = 0; t < keyTypes.size(); t++)
    {
        std::string ending = (t == keyTypes.size() - 1) ? ("}" + lineEnd) : ", ";
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
            std::cout << reinterpret_cast<int32_t*>(keys[t])[keyIndex] << ending;
            break;
        case DataType::COLUMN_LONG:
            std::cout << reinterpret_cast<int64_t*>(keys[t])[keyIndex] << ending;
            break;
        case DataType::COLUMN_FLOAT:
            std::cout << reinterpret_cast<float*>(keys[t])[keyIndex] << ending;
            break;
        case DataType::COLUMN_DOUBLE:
            std::cout << reinterpret_cast<double*>(keys[t])[keyIndex] << ending;
            break;
        case DataType::COLUMN_STRING:
            std::cout << "\"" << reinterpret_cast<std::string*>(keys[t])[keyIndex] << "\"" << ending;
            break;
        case DataType::COLUMN_INT8_T:
            std::cout << reinterpret_cast<int8_t*>(keys[t])[keyIndex] << ending;
            break;
        default:
            break;
        }
    }
}


template <typename AGG>
void TestGroupByMultiKey(std::vector<DataType> keyTypes,
                         std::vector<std::vector<void*>> keys,
                         std::vector<std::vector<int32_t>> values,
                         std::vector<void*> correctKeys,
                         std::vector<int32_t> correctValues)
{
    constexpr int32_t hashTableSize = 8;
    GPUGroupBy<AGG, int32_t, std::vector<void*>, int32_t> groupBy(hashTableSize, 1, keyTypes);
    int32_t keysColCount = keyTypes.size();
    for (int32_t b = 0; b < keys.size(); b++) // per "block"
    {
        // std::cout << "BLOCK " << b << ":" << std::endl;
        int32_t dataElementCount = values[b].size();
        std::vector<void*> gpuInKeys;
        std::vector<nullmask_t*> gpuInKeysNullMasks;
        for (int32_t t = 0; t < keysColCount; t++)
        {
            switch (keyTypes[t])
            {
            case DataType::COLUMN_INT:
            {
                int32_t* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, reinterpret_cast<int32_t*>(keys[b][t]), dataElementCount);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            case DataType::COLUMN_LONG:
            {
                int64_t* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, reinterpret_cast<int64_t*>(keys[b][t]), dataElementCount);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            case DataType::COLUMN_FLOAT:
            {
                float* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, reinterpret_cast<float*>(keys[b][t]), dataElementCount);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            case DataType::COLUMN_DOUBLE:
            {
                double* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, reinterpret_cast<double*>(keys[b][t]), dataElementCount);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            case DataType::COLUMN_STRING:
            {
                GPUMemory::GPUString* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, 1);
                std::string* cpuStrArray = reinterpret_cast<std::string*>(keys[b][t]);
                GPUMemory::GPUString cpuStructInKeys =
                    StringFactory::PrepareGPUString(cpuStrArray, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, &cpuStructInKeys, 1);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            case DataType::COLUMN_INT8_T:
            {
                int8_t* inKeysSingleCol;
                GPUMemory::alloc(&inKeysSingleCol, dataElementCount);
                GPUMemory::copyHostToDevice(inKeysSingleCol, reinterpret_cast<int8_t*>(keys[b][t]), dataElementCount);
                gpuInKeys.emplace_back(inKeysSingleCol);
                break;
            }
            default:
                break;
            }
            gpuInKeysNullMasks.emplace_back(nullptr);
        }
        cuda_ptr<int32_t> gpuInValues(dataElementCount);
        GPUMemory::copyHostToDevice(gpuInValues.get(), values[b].data(), dataElementCount);

        groupBy.ProcessBlock(gpuInKeys, gpuInKeysNullMasks, gpuInValues.get(), dataElementCount, nullptr);
        for (int32_t t = 0; t < keysColCount; t++)
        {
            if (keyTypes[t] == DataType::COLUMN_STRING)
            {
                GPUMemory::GPUString cpuStruct;
                GPUMemory::copyDeviceToHost(&cpuStruct,
                                            reinterpret_cast<GPUMemory::GPUString*>(gpuInKeys[t]), 1);
                GPUMemory::free(cpuStruct);
            }
            GPUMemory::free(gpuInKeys[t]);
        }
    }
    std::vector<void*> gpuResultKeys;
    int32_t* resultValuesGpu;
    int32_t resultCount;
    groupBy.GetResults(&gpuResultKeys, &resultValuesGpu, &resultCount);
    std::vector<void*> cpuResultKeys;
    for (int32_t t = 0; t < keysColCount; t++)
    {
        switch (keyTypes[t])
        {
        case DataType::COLUMN_INT:
        {
            int32_t* outKeysSingleCol = new int32_t[resultCount];
            GPUMemory::copyDeviceToHost(outKeysSingleCol, reinterpret_cast<int32_t*>(gpuResultKeys[t]), resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        case DataType::COLUMN_LONG:
        {
            int64_t* outKeysSingleCol = new int64_t[resultCount];
            GPUMemory::copyDeviceToHost(outKeysSingleCol, reinterpret_cast<int64_t*>(gpuResultKeys[t]), resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        case DataType::COLUMN_FLOAT:
        {
            float* outKeysSingleCol = new float[resultCount];
            GPUMemory::copyDeviceToHost(outKeysSingleCol, reinterpret_cast<float*>(gpuResultKeys[t]), resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        case DataType::COLUMN_DOUBLE:
        {
            double* outKeysSingleCol = new double[resultCount];
            GPUMemory::copyDeviceToHost(outKeysSingleCol, reinterpret_cast<double*>(gpuResultKeys[t]), resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        case DataType::COLUMN_STRING:
        {
            std::string* outKeysSingleCol = new std::string[resultCount];
            GPUReconstruct::ReconstructStringCol(outKeysSingleCol, &resultCount,
                                                 *reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[t]),
                                                 nullptr, resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        case DataType::COLUMN_INT8_T:
        {
            int8_t* outKeysSingleCol = new int8_t[resultCount];
            GPUMemory::copyDeviceToHost(outKeysSingleCol, reinterpret_cast<int8_t*>(gpuResultKeys[t]), resultCount);
            cpuResultKeys.emplace_back(outKeysSingleCol);
            break;
        }
        default:
            break;
        }
    }

    std::unique_ptr<int32_t[]> resultValues = std::make_unique<int32_t[]>(resultCount);
    GPUMemory::copyDeviceToHost(resultValues.get(), resultValuesGpu, resultCount);

    for (int32_t t = 0; t < keysColCount; t++)
    {
        if (keyTypes[t] == DataType::COLUMN_STRING)
        {
            GPUMemory::free(*reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[t]));
            delete reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[t]);
        }
        else
        {
            GPUMemory::free(gpuResultKeys[t]);
        }
    }
    GPUMemory::free(resultValuesGpu);

    ASSERT_EQ(correctValues.size(), resultCount) << " wrong number of keys";
    for (int32_t i = 0; i < resultCount; i++)
    {
        int32_t rowId = -1;
        for (int32_t j = 0; j < resultCount; j++)
        {
            bool equals = true;
            for (int32_t t = 0; t < keysColCount; t++)
            {
                switch (keyTypes[t])
                {
                case DataType::COLUMN_INT:
                    equals &= (reinterpret_cast<int32_t*>(correctKeys[t])[j] ==
                               reinterpret_cast<int32_t*>(cpuResultKeys[t])[i]);
                    break;
                case DataType::COLUMN_LONG:
                    equals &= (reinterpret_cast<int64_t*>(correctKeys[t])[j] ==
                               reinterpret_cast<int64_t*>(cpuResultKeys[t])[i]);
                    break;
                case DataType::COLUMN_FLOAT:
                    equals &= (reinterpret_cast<float*>(correctKeys[t])[j] ==
                               reinterpret_cast<float*>(cpuResultKeys[t])[i]);
                    break;
                case DataType::COLUMN_DOUBLE:
                    equals &= (reinterpret_cast<double*>(correctKeys[t])[j] ==
                               reinterpret_cast<double*>(cpuResultKeys[t])[i]);
                    break;
                case DataType::COLUMN_STRING:
                    equals &= (reinterpret_cast<std::string*>(correctKeys[t])[j] ==
                               reinterpret_cast<std::string*>(cpuResultKeys[t])[i]);
                    break;
                case DataType::COLUMN_INT8_T:
                    equals &= (reinterpret_cast<int8_t*>(correctKeys[t])[j] ==
                               reinterpret_cast<int8_t*>(cpuResultKeys[t])[i]);
                    break;
                default:
                    break;
                }
            }
            if (equals)
            {
                rowId = j;
                break;
            }
        }
        DebugPrintMultiKey(keyTypes, cpuResultKeys, i, ", ");
        std::cout << "value: " << resultValues[i] << std::endl;

        ASSERT_NE(rowId, -1) << " incorrect key";
        ASSERT_EQ(correctValues[rowId], resultValues[i]) << " at correct result row " << rowId;
    }
    /* // not needed free-ing (memory automatically frees everything after the test)
        for (int32_t t = 0; t < keysColCount; t++)
        {
            delete[] cpuResultKeys[t];
        }
    */
}

template <typename AGG>
void TestGroupByMultiKeyIntString(int32_t totalElementCount)
{
    int32_t blocks = 2;
    int32_t dataElementCount = totalElementCount / blocks;
    const int32_t intKey = 5;
    std::string strKey = "yellow";
    const float intVal = 1.0f;

    // std::vector<DataType> keyTypes, std::vector<std::vector<void*>> keys
    // std::vector<std::vector<int32_t>> values,
    // std::vector<void*> correctKeys, std::vector<int32_t> correctValues
    std::vector<DataType> keyTypes = {DataType::COLUMN_INT, DataType::COLUMN_STRING};
    constexpr int32_t hashTableSize = 65536;
    GPUGroupBy<AGG, float, std::vector<void*>, float> groupBy(hashTableSize, 1, keyTypes);
    int32_t keysColCount = keyTypes.size();
    for (int32_t b = 0; b < blocks; b++) // per "block"
    {
        std::vector<void*> gpuInKeys;
        std::vector<nullmask_t*> gpuInKeysNullMasks{nullptr, nullptr};

        int32_t* inIntKeysSingleCol;
        GPUMemory::alloc(&inIntKeysSingleCol, dataElementCount);
        GPUMemory::fillArray(inIntKeysSingleCol, intKey, dataElementCount);
        gpuInKeys.emplace_back(inIntKeysSingleCol);

        GPUMemory::GPUString* inStrKeySingleCol;
        GPUMemory::alloc(&inStrKeySingleCol, 1);
        std::vector<std::string> cpuString(dataElementCount, strKey);
        GPUMemory::GPUString cpuStructInKeys = StringFactory::PrepareGPUString(cpuString.data(), cpuString.size());
        GPUMemory::copyHostToDevice(inStrKeySingleCol, &cpuStructInKeys, 1);
        gpuInKeys.emplace_back(inStrKeySingleCol);

        cuda_ptr<float> gpuInValues(dataElementCount);
        GPUMemory::fillArray(gpuInValues.get(), intVal, dataElementCount);

        groupBy.ProcessBlock(gpuInKeys, gpuInKeysNullMasks, gpuInValues.get(), dataElementCount, nullptr);
        for (int32_t t = 0; t < keysColCount; t++)
        {
            if (keyTypes[t] == DataType::COLUMN_STRING)
            {
                GPUMemory::GPUString cpuStruct;
                GPUMemory::copyDeviceToHost(&cpuStruct,
                                            reinterpret_cast<GPUMemory::GPUString*>(gpuInKeys[t]), 1);
                GPUMemory::free(cpuStruct);
            }
            GPUMemory::free(gpuInKeys[t]);
        }
    }
    std::vector<void*> gpuResultKeys;
    float* resultValuesGpu;
    int32_t resultCount;
    groupBy.GetResults(&gpuResultKeys, &resultValuesGpu, &resultCount);

    ASSERT_EQ(1, resultCount);

    std::unique_ptr<int32_t[]> outIntKeySingleCol = std::make_unique<int32_t[]>(resultCount);
    GPUMemory::copyDeviceToHost(outIntKeySingleCol.get(), reinterpret_cast<int32_t*>(gpuResultKeys[0]), resultCount);
    ASSERT_EQ(intKey, outIntKeySingleCol[0]);

    std::unique_ptr<std::string[]> outStringKeySingleCol = std::make_unique<std::string[]>(resultCount);
    GPUReconstruct::ReconstructStringCol(outStringKeySingleCol.get(), &resultCount,
                                         *reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[1]),
                                         nullptr, resultCount);
    ASSERT_EQ(strKey, outStringKeySingleCol[0]);

    std::unique_ptr<float[]> resultValues = std::make_unique<float[]>(resultCount);
    GPUMemory::copyDeviceToHost(resultValues.get(), resultValuesGpu, resultCount);
    if (std::is_same<AGG, AggregationFunctions::sum>::value)
    {
        ASSERT_EQ(intVal * dataElementCount, resultValues[0]);
    }
    else if (std::is_same<AGG, AggregationFunctions::avg>::value)
    {
        ASSERT_EQ(intVal, resultValues[0]);
    }
    else
    {
        FAIL();
    }

    for (int32_t t = 0; t < keysColCount; t++)
    {
        if (keyTypes[t] == DataType::COLUMN_STRING)
        {
            GPUMemory::free(*reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[t]));
            delete reinterpret_cast<GPUMemory::GPUString*>(gpuResultKeys[t]);
        }
        else
        {
            GPUMemory::free(gpuResultKeys[t]);
        }
    }
    GPUMemory::free(resultValuesGpu);
}

void TestGroupByNoAgg(std::vector<std::vector<int32_t>> inKeys)
{
    // Create gb object
    GPUGroupBy<AggregationFunctions::none, int32_t, int32_t, int32_t> groupByObject(256, 1);
    const size_t blockCount = inKeys.size();
    std::set<int32_t> correctOutKeys;
    // Process all blocks
    for (size_t i = 0; i < blockCount; i++)
    {
        const size_t inDataElementCount = inKeys[i].size();
        cuda_ptr<int32_t> d_inKeys(inDataElementCount);
        GPUMemory::copyHostToDevice(d_inKeys.get(), inKeys[i].data(), inDataElementCount);
        groupByObject.ProcessBlock(d_inKeys.get(), nullptr, inDataElementCount);
        // Collect correct keys
        for (int32_t value : inKeys[i])
        {
            correctOutKeys.insert(value);
        }
    }
    // Return results
    int32_t outDataElementCount;
    int32_t* d_outKeys;
    groupByObject.GetResults(&d_outKeys, nullptr, &outDataElementCount);
    std::vector<int32_t> outKeys(outDataElementCount);
    GPUMemory::copyDeviceToHost(outKeys.data(), d_outKeys, outDataElementCount);
    GPUMemory::free(d_outKeys);
    // Print keys
    std::cout << "Correct: ";
    for (int32_t value : correctOutKeys)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl << "Actual: ";
    for (int32_t value : outKeys)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    // Assert keys
    ASSERT_EQ(correctOutKeys.size(), outDataElementCount);
    for (int32_t value : outKeys)
    {
        ASSERT_FALSE(correctOutKeys.find(value) == correctOutKeys.end());
    }
}


//== Group By String ==
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


//== Multi-Key Int tests ==
TEST(GPUGroupByTests, MultiKeyUnique)
{
    int32_t colA[] = {1, 1, 1, 1, 2, 3, 4, 0};
    int32_t colB[] = {1, 2, 3, 4, 1, 1, 1, 0};
    int32_t correctKeysA[] = {1, 1, 1, 1, 2, 3, 4, 0};
    int32_t correctKeysB[] = {1, 2, 3, 4, 1, 1, 1, 0};
    TestGroupByMultiKey<AggregationFunctions::sum>({DataType::COLUMN_INT, DataType::COLUMN_INT},
                                                   {{colA, colB}}, {{1, 1, 1, 1, 1, 1, 1, 1}},
                                                   {correctKeysA, correctKeysB}, {1, 1, 1, 1, 1, 1, 1, 1});
}

TEST(GPUGroupByTests, MultiKeySimple)
{
    int32_t colA[] = {1, 1, 1, 2, 1, 1, 2, 2};
    int32_t colB[] = {1, 2, 3, 4, 1, 1, -1, -1};
    int32_t correctKeysA[] = {1, 1, 1, 2, 2};
    int32_t correctKeysB[] = {1, 2, 3, 4, -1};
    TestGroupByMultiKey<AggregationFunctions::sum>({DataType::COLUMN_INT, DataType::COLUMN_INT},
                                                   {{colA, colB}}, {{1, 1, 1, 1, 1, 1, 1, 1}},
                                                   {correctKeysA, correctKeysB}, {3, 1, 1, 1, 2});
}


//== Multi-Key Int and String tests ==
TEST(GPUGroupByTests, MultiKeyStringSimpleSum)
{
    int32_t colA[] = {5, 2, 2, 2, 2, 5, 1, 7};
    int32_t colB[] = {1, 1, 1, 1, 1, 1, 2, 0};
    std::string colC[] = {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"};
    std::vector<int32_t> values = {1, 1, 1, 1, 1, 1, 1, 1};

    int32_t correctKeysA[] = {2, 2, 1, 7, 5, 2};
    int32_t correctKeysB[] = {1, 1, 2, 0, 1, 1};
    std::string correctKeysC[] = {"Apple", "XYZ", "Apple", "Nut", "Apple", "Nut"};
    std::vector<int32_t> correctValues = {1, 1, 1, 1, 2, 2};

    TestGroupByMultiKey<AggregationFunctions::sum>({DataType::COLUMN_INT, DataType::COLUMN_INT, DataType::COLUMN_STRING},
                                                   {{colA, colB, colC}}, {values},
                                                   {correctKeysA, correctKeysB, correctKeysC}, correctValues);
}

TEST(GPUGroupByTests, MultiKeyStringLargeDataSum)
{
    int32_t colA[] = {5, 2, 2, 2, 2, 5, 1, 7};
    int32_t colB[] = {1, 1, 1, 1, 1, 1, 2, 0};
    std::string colC[] = {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"};
    std::vector<int32_t> values = {1, 1, 1, 1, 1, 1, 1, 1};

    int32_t correctKeysA[] = {2, 2, 1, 7, 5, 2};
    int32_t correctKeysB[] = {1, 1, 2, 0, 1, 1};
    std::string correctKeysC[] = {"Apple", "XYZ", "Apple", "Nut", "Apple", "Nut"};
    std::vector<int32_t> correctValues = {1, 1, 1, 1, 2, 2};

    TestGroupByMultiKey<AggregationFunctions::sum>({DataType::COLUMN_INT, DataType::COLUMN_INT, DataType::COLUMN_STRING},
                                                   {{colA, colB, colC}}, {values},
                                                   {correctKeysA, correctKeysB, correctKeysC}, correctValues);
}

TEST(GPUGroupByTests, MultiKeyStringSimpleMin)
{
    int32_t colA[] = {8, 2, 2, 8, 2, 8, 1, 7};
    int32_t colB[] = {1, 1, 1, 1, 1, 1, 2, 0};
    std::string colC[] = {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"};
    std::vector<int32_t> values = {-1, 3, 70, 1, 1, 9, 2, 1};

    int32_t correctKeysA[] = {8, 2, 1, 7, 2};
    int32_t correctKeysB[] = {1, 1, 2, 0, 1};
    std::string correctKeysC[] = {"Apple", "XYZ", "Apple", "Nut", "Nut"};
    std::vector<int32_t> correctValues = {-1, 1, 2, 1, 3};

    TestGroupByMultiKey<AggregationFunctions::min>({DataType::COLUMN_INT, DataType::COLUMN_INT, DataType::COLUMN_STRING},
                                                   {{colA, colB, colC}}, {values},
                                                   {correctKeysA, correctKeysB, correctKeysC}, correctValues);
}

TEST(GPUGroupByTests, MultiKeyStringSimpleMax)
{
    int32_t colA[] = {8, 2, 2, 8, 2, 8, 1, 7};
    int32_t colB[] = {1, 1, 1, 1, 1, 1, 2, 0};
    std::string colC[] = {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"};
    std::vector<int32_t> values = {-1, 3, 70, 1, 1, 9, 2, 1};

    int32_t correctKeysA[] = {8, 2, 1, 7, 2};
    int32_t correctKeysB[] = {1, 1, 2, 0, 1};
    std::string correctKeysC[] = {"Apple", "XYZ", "Apple", "Nut", "Nut"};
    std::vector<int32_t> correctValues = {9, 1, 2, 1, 70};

    TestGroupByMultiKey<AggregationFunctions::max>({DataType::COLUMN_INT, DataType::COLUMN_INT, DataType::COLUMN_STRING},
                                                   {{colA, colB, colC}}, {values},
                                                   {correctKeysA, correctKeysB, correctKeysC}, correctValues);
}

TEST(GPUGroupByTests, MultiKeyIntStringAvgBig)
{
    TestGroupByMultiKeyIntString<AggregationFunctions::avg>(1 << 16);
    // TODO: for task #16666 in TP use greater value, e.g. 20000000 (16 777 216 (1<<24) is the edge)
}


//== Group By without Aggregation ==
TEST(GPUGroupByTests, IntKeyNoAggOneBlock)
{
    TestGroupByNoAgg({{0, 1, 2, 3, 10, 7, 1, 2, 5, -1, 3, 2, 1, 0, 0, 7}});
}

TEST(GPUGroupByTests, IntKeyNoAggMoreBlocks)
{
    TestGroupByNoAgg({{0, 1, 2, 3}, {0, 1, 2, 3}, {3, 1, 0, 2}, {1, 1, 3, -5}});
}
