#include <memory>
#include <cstdlib>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"

TEST(GPUFilterTests, FiltersColCol)
{
	//initialize random generators:
	const int32_t SEED = 42;
	const int32_t MIN_VALUE = -(1 << 10);
	const int32_t MAX_VALUE = 1 << 10;

	//set seed:
	srand(SEED);

	//initialize CUDA context:
	Context::getInstance();

	//size of data:
	const int32_t dataElementCount = 1 << 18;

	//CPU data:
	std::unique_ptr<int32_t[]> inputDataA = std::make_unique<int32_t[]>(dataElementCount);
	std::unique_ptr<int32_t[]> inputDataB = std::make_unique<int32_t[]>(dataElementCount);
	std::unique_ptr<int32_t[]> outputData = std::make_unique<int32_t[]>(dataElementCount);

	//fill input data buffers:
	for (int i = 0; i < dataElementCount; i++)
	{
		printf(" %d ", rand());
	}

	//create CUDA buffers:
	int32_t* inputBufferA;
	int32_t* inputBufferB;
	int32_t* outputBuffer;

	//alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, dataElementCount);
	GPUMemory::alloc(&inputBufferB, dataElementCount);
	GPUMemory::alloc(&outputBuffer, dataElementCount);

	ASSERT_EQ(true, true);

	//free buffers in GPU memory:
	GPUMemory::free(inputBufferA);
	GPUMemory::free(inputBufferB);
	GPUMemory::free(outputBuffer);
}
