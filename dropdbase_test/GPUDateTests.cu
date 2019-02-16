#include <memory>
#include <cstdlib>
#include <cstdint>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUDate.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"

const int32_t TEST_EL_COUNT = 8;
int64_t testDateTimes[] =        { 0, 1, 60, 3599, 3600, 5555, 86399, 86400, };
const int32_t correctYears[] =   { 0, 0,  0,    0,    0,    0,     0,     0, };
const int32_t correctMonths[] =  { 0, 0,  0,    0,    0,    0,     0,     0, };
const int32_t correctDays[] =    { 0, 0,  0,    0,    0,    0,     0,     1, };
const int32_t correctHours[] =   { 0, 0,  0,    0,    1,    1,    23,     0, };
const int32_t correctMinutes[] = { 0, 0,  1,   59,    0,   32,    59,     0, };
const int32_t correctSeconds[] = { 0, 1,  0,   59,    0,   35,    59,     0, };


TEST(GPUDateTests, ExtractHours)
{
	// Initialize CUDA context
	Context::getInstance();

	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(TEST_EL_COUNT);
	// Use our cuda smart pointers
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);
	cuda_ptr<int32_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUDate::extractCol<DateOperations::hour>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		ASSERT_EQ(resultHost.get()[i], correctHours[i]);
	}
}

TEST(GPUDateTests, ExtractMinutes)
{
	// Initialize CUDA context
	Context::getInstance();

	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(TEST_EL_COUNT);
	// Use our cuda smart pointers
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);
	cuda_ptr<int32_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUDate::extractCol<DateOperations::minute>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		ASSERT_EQ(resultHost.get()[i], correctMinutes[i]);
	}
}

TEST(GPUDateTests, ExtractSeconds)
{
	// Initialize CUDA context
	Context::getInstance();

	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(TEST_EL_COUNT);
	// Use our cuda smart pointers
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);
	cuda_ptr<int32_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUDate::extractCol<DateOperations::second>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		ASSERT_EQ(resultHost.get()[i], correctSeconds[i]);
	}
}
