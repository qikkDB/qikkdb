#include <memory>
#include <cstdlib>
#include <cstdio>
#include <random>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUFilter.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUFilterConditions.cuh"
#include "../dropdbase/StringFactory.h"


// Count of the testing data elements:
const int32_t DATA_ELEMENT_COUNT = 1 << 18;

// Float limits of random generator
constexpr float TEST_FLOAT_LOWEST = -32000.0f;
constexpr float TEST_FLOAT_HIGHEST = 32000.0f;

template <typename T>
void testColColFilter()
{
    // CPU data:
    std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
    std::unique_ptr<T[]> inputDataB = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
    std::unique_ptr<int8_t[]> outputData = std::make_unique<int8_t[]>(DATA_ELEMENT_COUNT);

    // Fill input data buffers:
    std::default_random_engine generator;
    if (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(),
                                                               std::numeric_limits<int32_t>::max());
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionInt(generator);
            inputDataB[i] = distributionInt(generator);
        }
    }
    else
    {
        std::uniform_real_distribution<float> distributionFloat(TEST_FLOAT_LOWEST, TEST_FLOAT_HIGHEST);
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionFloat(generator);
            inputDataB[i] = distributionFloat(generator);
        }
    }


    // Create CUDA buffers:
    T* inputBufferA;
    T* inputBufferB;
    int8_t* outputBuffer;

    // Alloc buffers in GPU memory:
    GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
    GPUMemory::alloc(&inputBufferB, DATA_ELEMENT_COUNT);
    GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

    // Copy the contents of the buffers to the GPU
    GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);
    GPUMemory::copyHostToDevice(inputBufferB, inputDataB.get(), DATA_ELEMENT_COUNT);

    //////////////////////////////////////////////////////////////////////////////////////
    // Run kernels, copy back values and compare them

    // Greater than
    GPUFilter::Filter<FilterConditions::greater>(outputBuffer, inputBufferA, inputBufferB, nullptr,
                                                 DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] > inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] > inputDataB[i]);
        }
    }

    // Greater than equal
    GPUFilter::Filter<FilterConditions::greaterEqual>(outputBuffer, inputBufferA, inputBufferB,
                                                      nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] >= inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] >= inputDataB[i]);
        }
    }

    // Less than
    GPUFilter::Filter<FilterConditions::less>(outputBuffer, inputBufferA, inputBufferB, nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] < inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] < inputDataB[i]);
        }
    }

    // Less than equal
    GPUFilter::Filter<FilterConditions::lessEqual>(outputBuffer, inputBufferA, inputBufferB,
                                                   nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] <= inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] <= inputDataB[i]);
        }
    }

    // Equal
    GPUFilter::Filter<FilterConditions::equal>(outputBuffer, inputBufferA, inputBufferB, nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] == inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] == inputDataB[i]);
        }
    }

    // Non equal
    GPUFilter::Filter<FilterConditions::notEqual>(outputBuffer, inputBufferA, inputBufferB, nullptr,
                                                  DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] != inputDataB[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] != inputDataB[i]);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////

    // Free buffers in GPU memory:
    GPUMemory::free(inputBufferA);
    GPUMemory::free(inputBufferB);
    GPUMemory::free(outputBuffer);
}


TEST(GPUFilterTests, FiltersColCol)
{
    // Initialize CUDA context:
    Context::getInstance();

    testColColFilter<int32_t>();
    testColColFilter<int64_t>();
    testColColFilter<float>();
}

//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void testColConstFilter()
{
    // CPU data:
    std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
    T inputDataBConstant;
    std::unique_ptr<int8_t[]> outputData = std::make_unique<int8_t[]>(DATA_ELEMENT_COUNT);

    // Fill input data buffers:
    std::default_random_engine generator;
    if (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(),
                                                               std::numeric_limits<int32_t>::max());
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionInt(generator);
        }
        inputDataBConstant = distributionInt(generator);
    }
    else
    {
        std::uniform_real_distribution<float> distributionFloat(TEST_FLOAT_LOWEST, TEST_FLOAT_HIGHEST);
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionFloat(generator);
        }
        inputDataBConstant = distributionFloat(generator);
    }


    // Create CUDA buffers:
    T* inputBufferA;
    int8_t* outputBuffer;

    // Alloc buffers in GPU memory:
    GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
    GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

    // Copy the contents of the buffers to the GPU
    GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

    //////////////////////////////////////////////////////////////////////////////////////
    // Run kernels, copy back values and compare them

    // Greater than
    GPUFilter::Filter<FilterConditions::greater>(outputBuffer, inputBufferA, inputDataBConstant,
                                                 nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] > inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] > inputDataBConstant);
        }
    }

    // Greater than equal
    GPUFilter::Filter<FilterConditions::greaterEqual>(outputBuffer, inputBufferA, inputDataBConstant,
                                                      nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] >= inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] >= inputDataBConstant);
        }
    }

    // Less than
    GPUFilter::Filter<FilterConditions::less>(outputBuffer, inputBufferA, inputDataBConstant,
                                              nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] < inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] < inputDataBConstant);
        }
    }

    // Less than equal
    GPUFilter::Filter<FilterConditions::lessEqual>(outputBuffer, inputBufferA, inputDataBConstant,
                                                   nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] <= inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] <= inputDataBConstant);
        }
    }

    // Equal
    GPUFilter::Filter<FilterConditions::equal>(outputBuffer, inputBufferA, inputDataBConstant,
                                               nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] == inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] == inputDataBConstant);
        }
    }

    // Non equal
    GPUFilter::Filter<FilterConditions::notEqual>(outputBuffer, inputBufferA, inputDataBConstant,
                                                  nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] != inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] != inputDataBConstant);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////

    // Free buffers in GPU memory:
    GPUMemory::free(inputBufferA);
    GPUMemory::free(outputBuffer);
}

TEST(GPUFilterTests, FiltersColConst)
{
    // Initialize CUDA context:
    Context::getInstance();

    testColConstFilter<int32_t>();
    testColConstFilter<int64_t>();
    testColConstFilter<float>();
}

//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void testConstColFilter()
{
    // CPU data:
    std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
    T inputDataBConstant;
    std::unique_ptr<int8_t[]> outputData = std::make_unique<int8_t[]>(DATA_ELEMENT_COUNT);

    // Fill input data buffers:
    std::default_random_engine generator;
    if (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(),
                                                               std::numeric_limits<int32_t>::max());
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionInt(generator);
        }
        inputDataBConstant = distributionInt(generator);
    }
    else
    {
        std::uniform_real_distribution<float> distributionFloat(TEST_FLOAT_LOWEST, TEST_FLOAT_HIGHEST);
        for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
        {
            inputDataA[i] = distributionFloat(generator);
        }
        inputDataBConstant = distributionFloat(generator);
    }


    // Create CUDA buffers:
    T* inputBufferA;
    int8_t* outputBuffer;

    // Alloc buffers in GPU memory:
    GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
    GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

    // Copy the contents of the buffers to the GPU
    GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

    //////////////////////////////////////////////////////////////////////////////////////
    // Run kernels, copy back values and compare them

    // Greater than
    GPUFilter::Filter<FilterConditions::greater>(outputBuffer, inputDataBConstant, inputBufferA,
                                                 nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataBConstant > inputDataA[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataBConstant > inputDataA[i]);
        }
    }

    // Greater than equal
    GPUFilter::Filter<FilterConditions::greaterEqual>(outputBuffer, inputDataBConstant,
                                                      inputBufferA, nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataBConstant >= inputDataA[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataBConstant >= inputDataA[i]);
        }
    }

    // Less than
    GPUFilter::Filter<FilterConditions::less>(outputBuffer, inputDataBConstant, inputBufferA,
                                              nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataBConstant < inputDataA[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataBConstant < inputDataA[i]);
        }
    }

    // Less than equal
    GPUFilter::Filter<FilterConditions::lessEqual>(outputBuffer, inputDataBConstant, inputBufferA,
                                                   nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataBConstant <= inputDataA[i]);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataBConstant <= inputDataA[i]);
        }
    }

    // Equal
    GPUFilter::Filter<FilterConditions::equal>(outputBuffer, inputDataBConstant, inputBufferA,
                                               nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] == inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] == inputDataBConstant);
        }
    }

    // Non equal
    GPUFilter::Filter<FilterConditions::notEqual>(outputBuffer, inputDataBConstant, inputBufferA,
                                                  nullptr, DATA_ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
    for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
    {
        if (std::is_integral<T>::value)
        {
            ASSERT_EQ(outputData[i], inputDataA[i] != inputDataBConstant);
        }
        else
        {
            ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] != inputDataBConstant);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////

    // Free buffers in GPU memory:
    GPUMemory::free(inputBufferA);
    GPUMemory::free(outputBuffer);
}

TEST(GPUFilterTests, FiltersConstCol)
{
    // Initialize CUDA context:
    Context::getInstance();

    testConstColFilter<int32_t>();
    testConstColFilter<int64_t>();
    testConstColFilter<float>();
}


//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void testConstConstFilter()
{
    // CPU data:
    T inputDataAConstant;
    T inputDataBConstant;
    int8_t outputData;

    // Fill input data buffers:
    std::default_random_engine generator;
    if (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(),
                                                               std::numeric_limits<int32_t>::max());

        inputDataAConstant = distributionInt(generator);
        inputDataBConstant = distributionInt(generator);
    }
    else
    {
        std::uniform_real_distribution<float> distributionFloat(TEST_FLOAT_LOWEST, TEST_FLOAT_HIGHEST);

        inputDataAConstant = distributionFloat(generator);
        inputDataBConstant = distributionFloat(generator);
    }

    // Create CUDA buffers:
    int8_t* outputBuffer;

    // Alloc buffers in GPU memory:
    GPUMemory::alloc(&outputBuffer, 1);

    //////////////////////////////////////////////////////////////////////////////////////
    // Run kernels, copy back values and compare them

    // Greater than
    GPUFilter::Filter<FilterConditions::greater>(outputBuffer, inputDataAConstant,
                                                 inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant > inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant > inputDataBConstant);
    }


    // Greater than equal
    GPUFilter::Filter<FilterConditions::greaterEqual>(outputBuffer, inputDataAConstant,
                                                      inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant >= inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant >= inputDataBConstant);
    }


    // Less than
    GPUFilter::Filter<FilterConditions::less>(outputBuffer, inputDataAConstant, inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant < inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant < inputDataBConstant);
    }


    // Less than equal
    GPUFilter::Filter<FilterConditions::lessEqual>(outputBuffer, inputDataAConstant,
                                                   inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant <= inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant <= inputDataBConstant);
    }


    // Equal
    GPUFilter::Filter<FilterConditions::equal>(outputBuffer, inputDataAConstant, inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant == inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant == inputDataBConstant);
    }


    // Non equal
    GPUFilter::Filter<FilterConditions::notEqual>(outputBuffer, inputDataAConstant,
                                                  inputDataBConstant, nullptr, 1);
    GPUMemory::copyDeviceToHost(&outputData, outputBuffer, 1);

    if (std::is_integral<T>::value)
    {
        ASSERT_EQ(outputData, inputDataAConstant != inputDataBConstant);
    }
    else
    {
        ASSERT_FLOAT_EQ(outputData, inputDataAConstant != inputDataBConstant);
    }


    //////////////////////////////////////////////////////////////////////////////////////

    // Free buffers in GPU memory:
    GPUMemory::free(outputBuffer);
}


TEST(GPUFilterTests, FiltersConstConst)
{
    // Initialize CUDA context:
    Context::getInstance();

    testConstConstFilter<int32_t>();
    testConstConstFilter<int64_t>();
    testConstConstFilter<float>();
}


// == String Filters ==
template <typename OP>
void TestFilterStringColCol(std::vector<std::string> inputStringACol,
                            std::vector<std::string> inputStringBCol,
                            std::vector<int8_t> expectedResults)
{
    GPUMemory::GPUString gpuStringACol =
        StringFactory::PrepareGPUString(inputStringACol.data(), inputStringACol.size());
    GPUMemory::GPUString gpuStringBCol =
        StringFactory::PrepareGPUString(inputStringBCol.data(), inputStringBCol.size());
    int32_t dataElementCount = std::min(inputStringACol.size(), inputStringBCol.size());
    cuda_ptr<int8_t> gpuMask(dataElementCount);
    GPUFilter::colCol<OP>(gpuMask.get(), gpuStringACol, gpuStringBCol, nullptr, dataElementCount);
    std::unique_ptr<int8_t[]> actualMask = std::make_unique<int8_t[]>(dataElementCount);
    GPUMemory::copyDeviceToHost(actualMask.get(), gpuMask.get(), dataElementCount);
    GPUMemory::free(gpuStringACol);
    GPUMemory::free(gpuStringBCol);

    ASSERT_EQ(dataElementCount, expectedResults.size());
    for (int32_t i = 0; i < dataElementCount; i++)
    {
        ASSERT_EQ(actualMask[i], expectedResults[i]) << " in ColCol at row " << i;
    }
}

template <typename OP>
void TestFilterStringColConst(std::vector<std::string> inputStringACol,
                              std::string inputStringBConst,
                              std::vector<int8_t> expectedResults)
{
    GPUMemory::GPUString gpuStringACol =
        StringFactory::PrepareGPUString(inputStringACol.data(), inputStringACol.size());
    GPUMemory::GPUString gpuStringBCol = StringFactory::PrepareGPUString(&inputStringBConst, 1);
    int32_t dataElementCount = inputStringACol.size();
    cuda_ptr<int8_t> gpuMask(dataElementCount);
    GPUFilter::colConst<OP>(gpuMask.get(), gpuStringACol, gpuStringBCol, nullptr, dataElementCount);
    std::unique_ptr<int8_t[]> actualMask = std::make_unique<int8_t[]>(dataElementCount);
    GPUMemory::copyDeviceToHost(actualMask.get(), gpuMask.get(), dataElementCount);
    GPUMemory::free(gpuStringACol);
    GPUMemory::free(gpuStringBCol);

    ASSERT_EQ(dataElementCount, expectedResults.size());
    for (int32_t i = 0; i < dataElementCount; i++)
    {
        ASSERT_EQ(actualMask[i], expectedResults[i]) << " in ColConst at row " << i;
    }
}

template <typename OP>
void TestFilterStringConstCol(std::string inputStringAConst,
                              std::vector<std::string> inputStringBCol,
                              std::vector<int8_t> expectedResults)
{
    GPUMemory::GPUString gpuStringACol = StringFactory::PrepareGPUString(&inputStringAConst, 1);
    GPUMemory::GPUString gpuStringBCol =
        StringFactory::PrepareGPUString(inputStringBCol.data(), inputStringBCol.size());
    int32_t dataElementCount = inputStringBCol.size();
    cuda_ptr<int8_t> gpuMask(dataElementCount);
    GPUFilter::constCol<OP>(gpuMask.get(), gpuStringACol, gpuStringBCol, nullptr, dataElementCount);
    std::unique_ptr<int8_t[]> actualMask = std::make_unique<int8_t[]>(dataElementCount);
    GPUMemory::copyDeviceToHost(actualMask.get(), gpuMask.get(), dataElementCount);
    GPUMemory::free(gpuStringACol);
    GPUMemory::free(gpuStringBCol);

    ASSERT_EQ(dataElementCount, expectedResults.size());
    for (int32_t i = 0; i < dataElementCount; i++)
    {
        ASSERT_EQ(actualMask[i], expectedResults[i]) << " in ConstCol at row " << i;
    }
}

template <typename OP>
void TestFilterStringConstConst(std::string inputStringAConst, std::string inputStringBConst, int8_t expectedResult)
{
    GPUMemory::GPUString gpuStringACol = StringFactory::PrepareGPUString(&inputStringAConst, 1);
    GPUMemory::GPUString gpuStringBCol = StringFactory::PrepareGPUString(&inputStringBConst, 1);
    int32_t dataElementCount = 8;
    cuda_ptr<int8_t> gpuMask(dataElementCount);
    GPUFilter::constConst<OP>(gpuMask.get(), gpuStringACol, gpuStringBCol, dataElementCount);
    std::unique_ptr<int8_t[]> actualMask = std::make_unique<int8_t[]>(dataElementCount);
    GPUMemory::copyDeviceToHost(actualMask.get(), gpuMask.get(), dataElementCount);
    GPUMemory::free(gpuStringACol);
    GPUMemory::free(gpuStringBCol);

    for (int32_t i = 0; i < dataElementCount; i++)
    {
        ASSERT_EQ(actualMask[i], expectedResult) << " in ConstConst at row " << i;
    }
}

TEST(GPUFilterTests, FiltersStringEq)
{
    TestFilterStringColCol<FilterConditions::equal>({"Abcd", "XYZW", " ", "_#$\\", "xpr"},
                                                    {"Abcd", "road", "z", "_#$\\", "x"}, {1, 0, 0, 1, 0});
    TestFilterStringColConst<FilterConditions::equal>({"Abcd", "XYZW", " ", "_#$\\", "xpr"}, "Abcd",
                                                      {1, 0, 0, 0, 0});
    TestFilterStringConstCol<FilterConditions::equal>("_#$\\", {"Abcd", "road", "z", "_#$\\", "x", ""},
                                                      {0, 0, 0, 1, 0, 0});
    TestFilterStringConstConst<FilterConditions::equal>("Abcd", "Abcd", 1);
    TestFilterStringConstConst<FilterConditions::equal>("Abcd", "road", 0);
}

TEST(GPUFilterTests, FiltersStringNonEq)
{
    TestFilterStringColCol<FilterConditions::notEqual>({"Abcd", "XYZW", " ", "_#$\\", "xpr"},
                                                       {"Abcd", "road", "zzz", "_#$\\", "x"},
                                                       {0, 1, 1, 0, 1});
    TestFilterStringColConst<FilterConditions::notEqual>({"Abcd", "XYZW", " ", "_#$\\", "xpr"},
                                                         "Abcd", {0, 1, 1, 1, 1});
    TestFilterStringConstCol<FilterConditions::notEqual>("_#$\\", {"Abcd", "road", "z", "_#$\\", "x", ""},
                                                         {1, 1, 1, 0, 1, 1});
    TestFilterStringConstConst<FilterConditions::notEqual>("Abcd", "Abcd", 0);
    TestFilterStringConstConst<FilterConditions::notEqual>("Abcd", "road", 1);
}
