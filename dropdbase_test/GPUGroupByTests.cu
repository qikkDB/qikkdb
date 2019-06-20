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

template<typename AGG>
void TestGroupByString(std::vector<std::string> keys, std::vector<int32_t> values)
{
	constexpr int32_t hashTableSize = 8;
	int32_t dataElementCount = min(keys.size(), values.size());
	GPUGroupBy<AGG, int32_t, GPUMemory::GPUString, int32_t> groupBy(hashTableSize);
	GPUMemory::GPUString gpuInKeys = StringFactory::PrepareGPUString(keys);
	cuda_ptr<int32_t> gpuInValues(dataElementCount);
	GPUMemory::copyHostToDevice(gpuInValues.get(), values.data(), dataElementCount);

	groupBy.groupBy(gpuInKeys, gpuInValues.get(), dataElementCount);
	int32_t sourceIds[hashTableSize];
	int32_t stringLens[hashTableSize];
	GPUMemory::copyDeviceToHost(sourceIds, groupBy.sourceIndices_, hashTableSize);
	GPUMemory::copyDeviceToHost(stringLens, groupBy.stringLengths_, hashTableSize);
	for (int32_t i = 0; i < hashTableSize; i++)
	{
		std::cout << stringLens[i] << "  " << sourceIds[i];
		if (sourceIds[i] >= 0)
		{
			std::cout << " (" << keys[sourceIds[i]] << ")";
		}
		std::cout << std::endl;
	}
	GPUMemory::free(gpuInKeys);
}


TEST(GPUGroupByTests, GroupByStringUnique)
{
	TestGroupByString<AggregationFunctions::sum>(
		{ "Apple", "Abcd", "XYZ"},
		{ 1, 2, -1 });
	FAIL();
}

TEST(GPUGroupByTests, GroupByStringSimple)
{
	TestGroupByString<AggregationFunctions::sum>(
		{ "Apple", "Abcd", "XYZ", "Abcd", "ZYX", "XYZ" },
		{ 1, 2, -1, 3, 7, 5 });
	FAIL();
}
