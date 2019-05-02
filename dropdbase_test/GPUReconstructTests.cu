#include <cstdint>
#include <cstdlib>
#include <memory>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"
#include "gtest/gtest.h"

template <typename M, typename T>
void TestGenerateIndexes(std::vector<M> mask, std::vector<T> correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = mask.size();
    cuda_ptr<M> maskDevice(ELEMENT_COUNT);
    std::unique_ptr<T[]> resultHost = std::make_unique<T[]>(ELEMENT_COUNT);
    int32_t outCount;

    GPUMemory::copyHostToDevice(maskDevice.get(), mask.data(), ELEMENT_COUNT);

    GPUReconstruct::GenerateIndexes(resultHost.get(), &outCount, maskDevice.get(), ELEMENT_COUNT);

    ASSERT_EQ(outCount, correctOutput.size()) << "incorrect output count";
    for (int i = 0; i < outCount; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput[i]) << "at [" << i << "], mask: " << mask[i];
    }
}

template <typename M, typename T>
void TestGenerateIndexesKeep(std::vector<M> mask, std::vector<T> correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = mask.size();
    cuda_ptr<M> maskDevice(ELEMENT_COUNT);
    T* resultDevice;
    int32_t outCount;

    GPUMemory::copyHostToDevice(maskDevice.get(), mask.data(), ELEMENT_COUNT);

    GPUReconstruct::GenerateIndexesKeep(&resultDevice, &outCount, maskDevice.get(), ELEMENT_COUNT);

    std::unique_ptr<T[]> resultHost = std::make_unique<T[]>(outCount);
    GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice, outCount);
    GPUMemory::free(resultDevice);

    ASSERT_EQ(outCount, correctOutput.size()) << "incorrect output count";
    for (int i = 0; i < outCount; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput[i]) << "at [" << i << "], mask: " << mask[i];
    }
}


TEST(GPUReconstructTests, GenerateIndexesFullMask)
{
    TestGenerateIndexes(std::vector<int8_t>{1, 1, 1, 1}, std::vector<int32_t>{0, 1, 2, 3});
}

TEST(GPUReconstructTests, GenerateIndexesEmptyMask)
{
    TestGenerateIndexes(std::vector<int8_t>{0, 0, 0, 0, 0, 0}, std::vector<int32_t>{});
}

TEST(GPUReconstructTests, GenerateIndexesMixMask)
{
    TestGenerateIndexes(std::vector<int8_t>{0, 1, 0, 0, 1, 1, 1, 1}, std::vector<int32_t>{1, 4, 5, 6, 7});
}

TEST(GPUReconstructTests, GenerateIndexesBigMask)
{
    std::vector<int8_t> mask;
    std::vector<int32_t> correctOutput;
    for (int32_t i = 0; i < (1 << 18); i++)
    {
        int8_t maskTrue = i % 2;
        mask.emplace_back(maskTrue);
        if (maskTrue)
        {
            correctOutput.emplace_back(i);
        }
    }
    TestGenerateIndexes(mask, correctOutput);
}

TEST(GPUReconstructTests, GenerateIndexesKeepMixMask)
{
    TestGenerateIndexesKeep(std::vector<int8_t>{0, 1, 0, 0, 1, 1, 1, 1}, std::vector<int32_t>{1, 4, 5, 6, 7});
}
