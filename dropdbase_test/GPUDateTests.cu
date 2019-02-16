
#include <memory>
#include <cstdlib>
#include <cstdint>


#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUFilter.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUDate.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"

const int32_t TEST_EL_COUNT = 12;
int64_t testDateTimes[] =      { 0, 1, 60, 3599, 3600, 5555, 86399, 86400, };
const int32_t correctYears[] = { 0, 0,  0,    0,    1,    1,    23,    24, };

//const int32_t correctMonths[] = {};
//const int32_t correctDays[] = {};
//const int32_t correctHours[] = {};
//const int32_t correctMinutes[] = {};
//const int32_t correctSeconds[] = {};


TEST(GPUDateTests, TestFilter) // len skuska ci z tohto testu ide napr. GPUFilter
{
	// Initialize CUDA context:
	Context::getInstance();

	std::unique_ptr<int8_t[]> resultHost = std::make_unique<int8_t[]>(TEST_EL_COUNT);
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);	// use our cuda smart pointer
	cuda_ptr<int8_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUFilter::colConst<FilterConditions::greater>(resultDevice.get(), dtDevice.get(), 3599, TEST_EL_COUNT);
	//GPUDate::extractCol<DateOperations::hour>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		ASSERT_EQ(resultHost.get()[i], static_cast<int8_t>(correctYears[i]));
	}
}

TEST(GPUDateTests, ExtractHour)
{
	// Initialize CUDA context:
	Context::getInstance();

	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(TEST_EL_COUNT);
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);	// use our cuda smart pointer
	cuda_ptr<int32_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUDate::extractCol<DateOperations::hour>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		ASSERT_EQ(resultHost.get()[i], correctYears[i]);
	}
}
