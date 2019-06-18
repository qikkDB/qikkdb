#include <memory>
#include <cstdlib>
#include <cstdio>
#include <random>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUFilter.cuh"


// Count of the testing data elements:
const int32_t DATA_ELEMENT_COUNT = 1 << 18;

// Float limits of random generator
constexpr float TEST_FLOAT_LOWEST = -32000.0f;
constexpr float TEST_FLOAT_HIGHEST = 32000.0f;

template<typename T>
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
		std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
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
	T *inputBufferA;
	T *inputBufferB;
	int8_t *outputBuffer;

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
	GPUFilter::colCol<FilterConditions::greater>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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
	GPUFilter::colCol<FilterConditions::greaterEqual>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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
	GPUFilter::colCol<FilterConditions::less>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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
	GPUFilter::colCol<FilterConditions::lessEqual>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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
	GPUFilter::colCol<FilterConditions::equal>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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
	GPUFilter::colCol<FilterConditions::notEqual>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
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

template<typename T>
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
		std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
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
	T *inputBufferA;
	int8_t *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Greater than
	GPUFilter::colConst<FilterConditions::greater>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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
	GPUFilter::colConst<FilterConditions::greaterEqual>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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
	GPUFilter::colConst<FilterConditions::less>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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
	GPUFilter::colConst<FilterConditions::lessEqual>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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
	GPUFilter::colConst<FilterConditions::equal>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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
	GPUFilter::colConst<FilterConditions::notEqual>(outputBuffer, inputBufferA, inputDataBConstant, DATA_ELEMENT_COUNT);
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

template<typename T>
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
		std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
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
	T *inputBufferA;
	int8_t *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Greater than
	GPUFilter::constCol<FilterConditions::greater>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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
	GPUFilter::constCol<FilterConditions::greaterEqual>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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
	GPUFilter::constCol<FilterConditions::less>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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
	GPUFilter::constCol<FilterConditions::lessEqual>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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
	GPUFilter::constCol<FilterConditions::equal>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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
	GPUFilter::constCol<FilterConditions::notEqual>(outputBuffer, inputDataBConstant, inputBufferA, DATA_ELEMENT_COUNT);
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

template<typename T>
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
		std::uniform_int_distribution<int32_t> distributionInt(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());

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
	int8_t *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&outputBuffer, 1);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Greater than
	GPUFilter::constConst<FilterConditions::greater>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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
	GPUFilter::constConst<FilterConditions::greaterEqual>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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
	GPUFilter::constConst<FilterConditions::less>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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
	GPUFilter::constConst<FilterConditions::lessEqual>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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
	GPUFilter::constConst<FilterConditions::equal>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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
	GPUFilter::constConst<FilterConditions::notEqual>(outputBuffer, inputDataAConstant, inputDataBConstant, 1);
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