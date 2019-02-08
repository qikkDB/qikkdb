#include <memory>
#include <cstdlib>
#include <cstdio>
#include <random>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../dropdbase/QueryEngine/GPUCore/AggregationFunctions.cuh"

// Initialize random generators with a seed
const int32_t SEED = 42;

// Count of the testing data elements:
const int32_t DATA_ELEMENT_COUNT = 1 << 18;

template<typename T>
void aggTests()
{
	// Alloc the buffers
	std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);

	T outputDataMin;
	T outputDataMax;
	T outputDataSum;
	T outputDataAvg;
	T outputDataCnt;

	// Fill input data buffers:
	std::default_random_engine generator;
	if (std::is_integral<T>::value)
	{
		std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionInt(generator);
		}
	}
	else
	{
		std::uniform_real_distribution<float> distributionFloat(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionFloat(generator);
		}
	}

	// Create CUDA buffers:
	T *inputBufferA;

	T *outputBufferMin;
	T *outputBufferMax;
	T *outputBufferSum;
	T *outputBufferAvg;
	T *outputBufferCnt;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);

	GPUMemory::alloc(&outputBufferMin, 1);
	GPUMemory::alloc(&outputBufferMax, 1);
	GPUMemory::alloc(&outputBufferSum, 1);
	GPUMemory::alloc(&outputBufferAvg, 1);
	GPUMemory::alloc(&outputBufferCnt, 1);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them
	GPUAggregation::col<AggregationFunctions::min>(outputBufferMin, inputBufferA, DATA_ELEMENT_COUNT);
	GPUAggregation::col<AggregationFunctions::max>(outputBufferMax, inputBufferA, DATA_ELEMENT_COUNT);
	GPUAggregation::col<AggregationFunctions::sum>(outputBufferSum, inputBufferA, DATA_ELEMENT_COUNT);
	GPUAggregation::col<AggregationFunctions::avg>(outputBufferAvg, inputBufferA, DATA_ELEMENT_COUNT);
	GPUAggregation::col<AggregationFunctions::count>(outputBufferCnt, inputBufferA, DATA_ELEMENT_COUNT);

	// Copy back data
	GPUMemory::copyDeviceToHost(&outputDataMin, outputBufferMin, 1);
	GPUMemory::copyDeviceToHost(&outputDataMax, outputBufferMax, 1);
	GPUMemory::copyDeviceToHost(&outputDataSum, outputBufferSum, 1);
	GPUMemory::copyDeviceToHost(&outputDataAvg, outputBufferAvg, 1);
	GPUMemory::copyDeviceToHost(&outputDataCnt, outputBufferCnt, 1);

	// Calculate the results on the cpu
	T min = std::numeric_limits<T>::max();
	T max = std::numeric_limits<T>::min();
	T sum = 0;
	T avg = 0;
	T cnt = 0;
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (min > inputDataA[i]) { min = inputDataA[i]; }
		if (max < inputDataA[i]) { max = inputDataA[i]; }

		sum += inputDataA[i];
	}
	avg = sum / DATA_ELEMENT_COUNT;
	cnt = DATA_ELEMENT_COUNT;

	// Check results
	if (std::is_integral<T>::value)
	{
		ASSERT_EQ(min, outputDataMin);
		ASSERT_EQ(max, outputDataMax);
		ASSERT_EQ(sum, outputDataSum);
		ASSERT_EQ(avg, outputDataAvg);
		ASSERT_EQ(cnt, outputDataCnt);
	}
	else
	{
		ASSERT_FLOAT_EQ(min, outputDataMin);
		ASSERT_FLOAT_EQ(max, outputDataMax);
		ASSERT_FLOAT_EQ(sum, outputDataSum);
		ASSERT_FLOAT_EQ(avg, outputDataAvg);
		ASSERT_FLOAT_EQ(cnt, outputDataCnt);
	}

	// Free the data
	GPUMemory::free(inputBufferA);

	GPUMemory::free(outputBufferMin);
	GPUMemory::free(outputBufferMax);
	GPUMemory::free(outputBufferSum);
	GPUMemory::free(outputBufferAvg);
	GPUMemory::free(outputBufferCnt);
}

TEST(GPUAggregationTests, AggTests)
{
	// Initialize CUDA context:
	Context::getInstance();

	aggTests<int32_t>();
	aggTests<int64_t>();
	aggTests<float>();
}