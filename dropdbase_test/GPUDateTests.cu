#include <memory>
#include <cstdlib>
#include <cstdint>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUDate.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"

std::vector<int64_t> testDateTimes =  {    0,    1,   59,   60,   61, 3599, 3600, 3601, 3661, 5555, 86399, 86400, 86401 };
std::vector<int32_t> correctYears =   { 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970,  1970,  1970,  1970 };
std::vector<int32_t> correctMonths =  {    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,     1,     1,     1 };
std::vector<int32_t> correctDays =    {    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,     1,     2,     2 };
std::vector<int32_t> correctHours =   {    0,    0,    0,    0,    0,    0,    1,    1,    1,    1,    23,     0,     0 };
std::vector<int32_t> correctMinutes = {    0,    0,    0,    1,    1,   59,    0,    0,    1,   32,    59,     0,     0 };
std::vector<int32_t> correctSeconds = {    0,    1,   59,    0,    1,   59,    0,    1,    1,   35,    59,     0,     1 };

std::vector<int64_t> testDateTimesNegative =  {    0,   -1,  -59,  -60,  -61, -3599, -3600, -5555, -86399, -86400, -86401 };
std::vector<int32_t> correctYearsNegative =   { 1970, 1969, 1969, 1969, 1969,  1969,  1969,  1969,   1969,   1969,   1969 };
std::vector<int32_t> correctMonthsNegative =  {    1,   12,   12,   12,   12,    12,    12,    12,     12,     12,     12 };
std::vector<int32_t> correctDaysNegative =    {    1,   31,   31,   31,   31,    31,    31,    31,     31,     31,     30 };
std::vector<int32_t> correctHoursNegative =   {    0,   23,   23,   23,   23,    23,    23,    22,      0,      0,     23 };
std::vector<int32_t> correctMinutesNegative = {    0,   59,   59,   59,   58,     0,     0,    27,      0,      0,     59 };
std::vector<int32_t> correctSecondsNegative = {    0,   59,   01,    0,   59,     1,     0,    25,      1,      0,     59 };

template<typename OP>
void testExtract(std::vector<int64_t>& input, std::vector<int32_t>& correctOutput)
{
	// Initialize CUDA context
	Context::getInstance();
	
	const int ELEMENT_COUNT = min(input.size(), correctOutput.size());
	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(ELEMENT_COUNT);
	// Use our cuda smart pointers
	cuda_ptr<int64_t> dtDevice(ELEMENT_COUNT);
	cuda_ptr<int32_t> resultDevice(ELEMENT_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), input.data(), ELEMENT_COUNT);
	GPUDate::extractCol<OP>(resultDevice.get(), dtDevice.get(), ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), ELEMENT_COUNT);

	for (int i = 0; i < ELEMENT_COUNT; i++)
	{
		std::cout << "asserting " << i << std::endl;
		ASSERT_EQ(resultHost.get()[i], correctOutput[i]);
	}
}

TEST(GPUDateTests, ExtractHours)
{
	testExtract<DateOperations::hour>(testDateTimes, correctHours);
}

TEST(GPUDateTests, ExtractMinutes)
{
	testExtract<DateOperations::minute>(testDateTimes, correctMinutes);
}

TEST(GPUDateTests, ExtractSeconds)
{
	testExtract<DateOperations::second>(testDateTimes, correctSeconds);
}

// Test negatite timestamps
TEST(GPUDateTests, ExtractHoursFromNegative)
{
	testExtract<DateOperations::hour>(testDateTimesNegative, correctHoursNegative);
}

TEST(GPUDateTests, ExtractMinutesFromNegative)
{
	testExtract<DateOperations::minute>(testDateTimesNegative, correctMinutesNegative);
}

TEST(GPUDateTests, ExtractSecondsFromNegative)
{
	testExtract<DateOperations::second>(testDateTimesNegative, correctSecondsNegative);
}
