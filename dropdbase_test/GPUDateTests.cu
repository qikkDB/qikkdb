#include <memory>
#include <cstdlib>
#include <cstdint>


#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUDate.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"

constexpr int32_t TEST_EL_COUNT = 12;
int64_t testDateTimes[] =          { 0, 1, 60, 3599, 3600, 5555, 86399, 86400, };
constexpr int32_t correctYears[] = { 0, 0,  0,    0,    1,    1,    23,    24, };
/*
constexpr int32_t correctMonths[] = {};
constexpr int32_t correctDays[] = {};
constexpr int32_t correctHours[] = {};
constexpr int32_t correctMinutes[] = {};
constexpr int32_t correctSeconds[] = {};
*/

TEST(GPUDateTests, ExtractionHour)
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
