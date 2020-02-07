#include <memory>
#include <cstdlib>
#include <cstdint>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUArithmeticUnary.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"

std::vector<int64_t> testDateTimes = {0,        1,        59,       60,       61,       3599,
                                      3600,     3601,     3661,     5555,     86399,    86400,
                                      86401,    172799,   172800,   2678399,  2678400,  5054400,
                                      5097600,  7776000,  10368000, 13046400, 15638400, 18316800,
                                      20995200, 23587200, 26265600, 31535999, 31536000, 31536001};
std::vector<int32_t> correctYears = {1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970,
                                     1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970,
                                     1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1971, 1971};
std::vector<int32_t> correctMonths = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1, 1,
                                      1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 1};
std::vector<int32_t> correctDays = {1,  1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 2, 2,  2, 3,
                                    31, 1, 28, 1, 1, 1, 1, 1, 1, 1, 1, 1, 31, 1, 1};
std::vector<int32_t> correctHours = {0,  0, 0,  0, 0, 0, 1, 1, 1, 1, 23, 0, 0,  23, 0,
                                     23, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0,  0, 23, 0,  0};
std::vector<int32_t> correctMinutes = {0,  0, 0, 1, 1, 59, 0, 0, 1, 32, 59, 0, 0,  59, 0,
                                       59, 0, 0, 0, 0, 0,  0, 0, 0, 0,  0,  0, 59, 0,  0};
std::vector<int32_t> correctSeconds = {0,  1, 59, 0, 1, 59, 0, 1, 1, 35, 59, 0, 1,  59, 0,
                                       59, 0, 0,  0, 0, 0,  0, 0, 0, 0,  0,  0, 59, 0,  1};

std::vector<int64_t> testDateTimesNegative = {0,     -1,    -59,    -60,    -61,   -3599,
                                              -3600, -5555, -86399, -86400, -86401};
std::vector<int32_t> correctYearsNegative = {1970, 1969, 1969, 1969, 1969, 1969,
                                             1969, 1969, 1969, 1969, 1969};
std::vector<int32_t> correctMonthsNegative = {1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
std::vector<int32_t> correctDaysNegative = {1, 31, 31, 31, 31, 31, 31, 31, 31, 31, 30};
std::vector<int32_t> correctHoursNegative = {0, 23, 23, 23, 23, 23, 23, 22, 0, 0, 23};
std::vector<int32_t> correctMinutesNegative = {0, 59, 59, 59, 58, 0, 0, 27, 0, 0, 59};
std::vector<int32_t> correctSecondsNegative = {0, 59, 01, 0, 59, 1, 0, 25, 1, 0, 59};

std::vector<int64_t> testDateTimesLeap = {
    5011200,       5097600,        36547200,       36633600,       68083200,       68169600,
    68256000,      99792000,       194486400,      825638400,      951782400,      951868800,
    1456790400,    1519862400,     1550501595,     1551398400,     1583020800,     3981398400LL,
    4107456000LL,  4107542399LL,   4107542400LL,   7263216000LL,   10418889600LL,  13574649600LL,
    64065772800LL, 253281254400LL, 253370764800LL, 253375862400LL, 253402214400LL, 253402300799LL};
std::vector<int32_t> correctYearsLeap = {1970, 1970, 1971, 1971, 1972, 1972, 1972, 1973,
                                         1976, 1996, 2000, 2000, 2016, 2018, 2019, 2019,
                                         2020, 2096, 2100, 2100, 2100, 2200, 2300, 2400,
                                         4000, 9996, 9999, 9999, 9999, 9999};
std::vector<int32_t> correctMonthsLeap = {2, 3, 2, 3, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3,  2,
                                          3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 1, 3, 12, 12};
std::vector<int32_t> correctDaysLeap = {28, 1, 28, 1,  28, 29, 1, 1, 1, 1, 29, 1, 1, 1,  18,
                                        1,  1, 1,  28, 28, 1,  1, 1, 1, 1, 1,  1, 1, 31, 31};
std::vector<int32_t> correctHoursLeap = {0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
                                         0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23};
std::vector<int32_t> correctMinutesLeap = {0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 53,
                                           0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59};
std::vector<int32_t> correctSecondsLeap = {0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 15,
                                           0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59};

std::vector<int64_t> testDateTimesLeapNegative = {-62135596800LL, -24293692800LL, -17982345600LL,
                                                  -11671084800LL, -11670998400LL, -11670912000LL,
                                                  -11639462400LL, -11639376000LL, -11544681600LL,
                                                  -8515238400LL,  -2865412800LL};
std::vector<int32_t> correctYearsLeapNegative = {1,    1200, 1400, 1600, 1600, 1600,
                                                 1601, 1601, 1604, 1700, 1879};
std::vector<int32_t> correctMonthsLeapNegative = {1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 3};
std::vector<int32_t> correctDaysLeapNegative = {1, 1, 1, 28, 29, 1, 28, 1, 1, 1, 14};
std::vector<int32_t> correctHoursLeapNegative = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12};
std::vector<int32_t> correctMinutesLeapNegative = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
std::vector<int32_t> correctSecondsLeapNegative = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


template <typename OP>
void testExtract(std::vector<int64_t>& input, std::vector<int32_t>& correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = static_cast<int>(min(input.size(), correctOutput.size()));
    std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(ELEMENT_COUNT);
    // Use our cuda smart pointers
    cuda_ptr<int64_t> dtDevice(ELEMENT_COUNT);
    cuda_ptr<int32_t> resultDevice(ELEMENT_COUNT);

    GPUMemory::copyHostToDevice(dtDevice.get(), input.data(), ELEMENT_COUNT);
    GPUArithmeticUnary::ArithmeticUnary<OP, int32_t, int64_t*>(resultDevice.get(), dtDevice.get(), ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), ELEMENT_COUNT);

    for (int i = 0; i < ELEMENT_COUNT; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput[i]) << " at [" << i << "], timestamp: " << input[i];
    }
}

template <typename OP>
void testExtractOneConstant(int64_t input, int32_t correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = 4;
    std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(ELEMENT_COUNT);
    // Use our cuda smart pointers
    cuda_ptr<int32_t> resultDevice(ELEMENT_COUNT);

    GPUArithmeticUnary::ArithmeticUnary<OP, int32_t, int64_t>(resultDevice.get(), input, ELEMENT_COUNT);
    GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), ELEMENT_COUNT);

    for (int i = 0; i < ELEMENT_COUNT; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput) << " timestamp: " << input;
    }
}

template <typename OP>
void testExtractAsConstants(std::vector<int64_t>& input, std::vector<int32_t>& correctOutput)
{
    const int ELEMENT_COUNT = static_cast<int>(min(input.size(), correctOutput.size()));

    for (int i = 0; i < ELEMENT_COUNT; i++)
    {
        testExtractOneConstant<OP>(input[i], correctOutput[i]);
    }
}


// Lightest tests - just some small timestamps (years 1970-1971)
TEST(GPUDateTests, ExtractYears)
{
    testExtract<DateOperations::year>(testDateTimes, correctYears);
}

TEST(GPUDateTests, ExtractMonths)
{
    testExtract<DateOperations::month>(testDateTimes, correctMonths);
}

TEST(GPUDateTests, ExtractDays)
{
    testExtract<DateOperations::day>(testDateTimes, correctDays);
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


// Test negative timestamps
TEST(GPUDateTests, ExtractYearsNegative)
{
    testExtract<DateOperations::year>(testDateTimesNegative, correctYearsNegative);
}

TEST(GPUDateTests, ExtractMonthsNegative)
{
    testExtract<DateOperations::month>(testDateTimesNegative, correctMonthsNegative);
}

TEST(GPUDateTests, ExtractDaysNegative)
{
    testExtract<DateOperations::day>(testDateTimesNegative, correctDaysNegative);
}

TEST(GPUDateTests, ExtractHoursNegative)
{
    testExtract<DateOperations::hour>(testDateTimesNegative, correctHoursNegative);
}

TEST(GPUDateTests, ExtractMinutesNegative)
{
    testExtract<DateOperations::minute>(testDateTimesNegative, correctMinutesNegative);
}

TEST(GPUDateTests, ExtractSecondsNegative)
{
    testExtract<DateOperations::second>(testDateTimesNegative, correctSecondsNegative);
}


// Test leap years etc.
TEST(GPUDateTests, ExtractYearsLeap)
{
    testExtract<DateOperations::year>(testDateTimesLeap, correctYearsLeap);
}

TEST(GPUDateTests, ExtractMonthsLeap)
{
    testExtract<DateOperations::month>(testDateTimesLeap, correctMonthsLeap);
}

TEST(GPUDateTests, ExtractDaysLeap)
{
    testExtract<DateOperations::day>(testDateTimesLeap, correctDaysLeap);
}

TEST(GPUDateTests, ExtractHoursLeap)
{
    testExtract<DateOperations::hour>(testDateTimesLeap, correctHoursLeap);
}

TEST(GPUDateTests, ExtractMinutesLeap)
{
    testExtract<DateOperations::minute>(testDateTimesLeap, correctMinutesLeap);
}

TEST(GPUDateTests, ExtractSecondsLeap)
{
    testExtract<DateOperations::second>(testDateTimesLeap, correctSecondsLeap);
}


// Test leap years at negative timestamps
TEST(GPUDateTests, ExtractYearsLeapNegative)
{
    testExtract<DateOperations::year>(testDateTimesLeapNegative, correctYearsLeapNegative);
}

TEST(GPUDateTests, ExtractMonthsLeapNegative)
{
    testExtract<DateOperations::month>(testDateTimesLeapNegative, correctMonthsLeapNegative);
}

TEST(GPUDateTests, ExtractDaysLeapNegative)
{
    testExtract<DateOperations::day>(testDateTimesLeapNegative, correctDaysLeapNegative);
}

TEST(GPUDateTests, ExtractHoursLeapNegative)
{
    testExtract<DateOperations::hour>(testDateTimesLeapNegative, correctHoursLeapNegative);
}

TEST(GPUDateTests, ExtractMinutesLeapNegative)
{
    testExtract<DateOperations::minute>(testDateTimesLeapNegative, correctMinutesLeapNegative);
}

TEST(GPUDateTests, ExtractSecondsLeapNegative)
{
    testExtract<DateOperations::second>(testDateTimesLeapNegative, correctSecondsLeapNegative);
}


// Versions with constant
TEST(GPUDateTests, ExtractYearConsts)
{
    testExtractAsConstants<DateOperations::year>(testDateTimes, correctYears);
    testExtractAsConstants<DateOperations::year>(testDateTimesNegative, correctYearsNegative);
    testExtractAsConstants<DateOperations::year>(testDateTimesLeap, correctYearsLeap);
    testExtractAsConstants<DateOperations::year>(testDateTimesLeapNegative, correctYearsLeapNegative);
}

TEST(GPUDateTests, ExtractMonthConsts)
{
    testExtractAsConstants<DateOperations::month>(testDateTimes, correctMonths);
    testExtractAsConstants<DateOperations::month>(testDateTimesNegative, correctMonthsNegative);
    testExtractAsConstants<DateOperations::month>(testDateTimesLeap, correctMonthsLeap);
    testExtractAsConstants<DateOperations::month>(testDateTimesLeapNegative, correctMonthsLeapNegative);
}

TEST(GPUDateTests, ExtractDayConsts)
{
    testExtractAsConstants<DateOperations::day>(testDateTimes, correctDays);
    testExtractAsConstants<DateOperations::day>(testDateTimesNegative, correctDaysNegative);
    testExtractAsConstants<DateOperations::day>(testDateTimesLeap, correctDaysLeap);
    testExtractAsConstants<DateOperations::day>(testDateTimesLeapNegative, correctDaysLeapNegative);
}

TEST(GPUDateTests, ExtractHourConsts)
{
    testExtractAsConstants<DateOperations::hour>(testDateTimes, correctHours);
    testExtractAsConstants<DateOperations::hour>(testDateTimesNegative, correctHoursNegative);
    testExtractAsConstants<DateOperations::hour>(testDateTimesLeap, correctHoursLeap);
    testExtractAsConstants<DateOperations::hour>(testDateTimesLeapNegative, correctHoursLeapNegative);
}

TEST(GPUDateTests, ExtractMinuteConsts)
{
    testExtractAsConstants<DateOperations::minute>(testDateTimes, correctMinutes);
    testExtractAsConstants<DateOperations::minute>(testDateTimesNegative, correctMinutesNegative);
    testExtractAsConstants<DateOperations::minute>(testDateTimesLeap, correctMinutesLeap);
    testExtractAsConstants<DateOperations::minute>(testDateTimesLeapNegative, correctMinutesLeapNegative);
}

TEST(GPUDateTests, ExtractSecondConsts)
{
    testExtractAsConstants<DateOperations::second>(testDateTimes, correctSeconds);
    testExtractAsConstants<DateOperations::second>(testDateTimesNegative, correctSecondsNegative);
    testExtractAsConstants<DateOperations::second>(testDateTimesLeap, correctSecondsLeap);
    testExtractAsConstants<DateOperations::second>(testDateTimesLeapNegative, correctSecondsLeapNegative);
}
