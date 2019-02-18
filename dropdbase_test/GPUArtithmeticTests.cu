#include <memory>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <cmath>

#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUArithmetic.cuh"

// Initialize random generators with a seed
const int32_t SEED = 42;

// Count of the testing data elements:
const int32_t DATA_ELEMENT_COUNT = 1 << 18;

template<typename T>
void testColColArithmetic()
{
	// CPU data:
	std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
	std::unique_ptr<T[]> inputDataB = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
	std::unique_ptr<T[]> outputData = std::make_unique<T[]>(DATA_ELEMENT_COUNT);

	// Fill input data buffers:
	std::default_random_engine generator;
	if (std::is_integral<T>::value)
	{
		std::uniform_int_distribution<int32_t> distributionInt(-1024, 1024);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionInt(generator);
			inputDataB[i] = distributionInt(generator);
		}
	}
	else
	{
		std::uniform_real_distribution<float> distributionFloat(-1024.0f, 1024.0f);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionFloat(generator);
			inputDataB[i] = distributionFloat(generator);
		}
	}


	// Create CUDA buffers:
	T *inputBufferA;
	T *inputBufferB;
	T *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);
	GPUMemory::copyHostToDevice(inputBufferB, inputDataB.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Add
	GPUArithmetic::colCol<ArithmeticOperations::add>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] + inputDataB[i]);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] + inputDataB[i]);
		}
	}

	// Sub
	GPUArithmetic::colCol<ArithmeticOperations::sub>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] - inputDataB[i]);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] - inputDataB[i]);
		}
	}

	// Mul
	GPUArithmetic::colCol<ArithmeticOperations::mul>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] * inputDataB[i]);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] * inputDataB[i]);
		}
	}

	// Floor div
	GPUArithmetic::colCol<ArithmeticOperations::floorDiv>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);

	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			if (inputDataB[i] != 0)
			{
				ASSERT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataB[i]));
			}
		}
		else
		{
			if (inputDataB[i] != 0)
			{
				ASSERT_FLOAT_EQ(outputData[i], static_cast<T>(floor(inputDataA[i] / inputDataB[i])));
			}
		}
	}

	// Div
	GPUArithmetic::colCol<ArithmeticOperations::div>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			if (inputDataB[i] != 0)
			{
				ASSERT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataB[i]));
			}
		}
		else
		{
			if (inputDataB[i] != 0)
			{
				ASSERT_FLOAT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataB[i]));
			}
		}
	}

	// Modulus
	if (std::is_integral<T>::value)
	{
		GPUArithmetic::colCol<ArithmeticOperations::mod>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
		GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			if (inputDataB[i] != 0)
			{
				ASSERT_EQ(outputData[i], inputDataA[i] % inputDataB[i]);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////

	// Free buffers in GPU memory:
	GPUMemory::free(inputBufferA);
	GPUMemory::free(inputBufferB);
	GPUMemory::free(outputBuffer);
}

template<>
void testColColArithmetic<float>()
{
	// CPU data:
	std::unique_ptr<float[]> inputDataA = std::make_unique<float[]>(DATA_ELEMENT_COUNT);
	std::unique_ptr<float[]> inputDataB = std::make_unique<float[]>(DATA_ELEMENT_COUNT);
	std::unique_ptr<float[]> outputData = std::make_unique<float[]>(DATA_ELEMENT_COUNT);

	// Fill input data buffers:
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distributionFloat(-1024.0f, 1024.0f);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		inputDataA[i] = distributionFloat(generator);
		inputDataB[i] = distributionFloat(generator);
	}

	// Create CUDA buffers:
	float *inputBufferA;
	float *inputBufferB;
	float *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);
	GPUMemory::copyHostToDevice(inputBufferB, inputDataB.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Add
	GPUArithmetic::colCol<ArithmeticOperations::add>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] + inputDataB[i]);
	}

	// Sub
	GPUArithmetic::colCol<ArithmeticOperations::sub>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] - inputDataB[i]);
	}

	// Mul
	GPUArithmetic::colCol<ArithmeticOperations::mul>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] * inputDataB[i]);
	}

	// Floor div
	GPUArithmetic::colCol<ArithmeticOperations::floorDiv>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);

	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (inputDataB[i] != 0)
		{
			ASSERT_FLOAT_EQ(outputData[i], static_cast<float>(floor(inputDataA[i] / inputDataB[i])));
		}
	}

	// Div
	GPUArithmetic::colCol<ArithmeticOperations::div>(outputBuffer, inputBufferA, inputBufferB, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (inputDataB[i] != 0)
		{
			ASSERT_FLOAT_EQ(outputData[i], static_cast<float>(inputDataA[i] / inputDataB[i]));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////

	// Free buffers in GPU memory:
	GPUMemory::free(inputBufferA);
	GPUMemory::free(inputBufferB);
	GPUMemory::free(outputBuffer);
}

TEST(GPUArithmeticTests, ArithmeticsColCol)
{
	// Initialize CUDA context:
	Context::getInstance();

	testColColArithmetic <int32_t>();
	testColColArithmetic <int64_t>();
	testColColArithmetic <float>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void testColConstArithmetic()
{
	// CPU data:
	std::unique_ptr<T[]> inputDataA = std::make_unique<T[]>(DATA_ELEMENT_COUNT);
	T inputDataBConst;
	std::unique_ptr<T[]> outputData = std::make_unique<T[]>(DATA_ELEMENT_COUNT);

	// Fill input data buffers:
	std::default_random_engine generator;
	if (std::is_integral<T>::value)
	{
		std::uniform_int_distribution<int32_t> distributionInt(-1024, 1024);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionInt(generator);
		}
		inputDataBConst = distributionInt(generator);
	}
	else
	{
		std::uniform_real_distribution<float> distributionFloat(-1024.0f, 1024.0f);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionFloat(generator);
		}
		inputDataBConst = distributionFloat(generator);
	}


	// Create CUDA buffers:
	T *inputBufferA;
	T *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Add
	GPUArithmetic::colConst<ArithmeticOperations::add>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] + inputDataBConst);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] + inputDataBConst);
		}
	}

	// Sub
	GPUArithmetic::colConst<ArithmeticOperations::sub>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] - inputDataBConst);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] - inputDataBConst);
		}
	}

	// Mul
	GPUArithmetic::colConst<ArithmeticOperations::mul>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			ASSERT_EQ(outputData[i], inputDataA[i] * inputDataBConst);
		}
		else
		{
			ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] * inputDataBConst);
		}
	}

	// Floor div
	GPUArithmetic::colConst<ArithmeticOperations::floorDiv>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);

	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			if (inputDataBConst != 0)
			{
				ASSERT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataBConst));
			}
		}
		else
		{
			if (inputDataBConst != 0)
			{
				ASSERT_FLOAT_EQ(outputData[i], static_cast<T>(floor(inputDataA[i] / inputDataBConst)));
			}
		}
	}

	// Div
	GPUArithmetic::colConst<ArithmeticOperations::div>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (std::is_integral<T>::value)
		{
			if (inputDataBConst != 0)
			{
				ASSERT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataBConst));
			}
		}
		else
		{
			if (inputDataBConst != 0)
			{
				ASSERT_FLOAT_EQ(outputData[i], static_cast<T>(inputDataA[i] / inputDataBConst));
			}
		}
	}

	// Modulus
	if (std::is_integral<T>::value)
	{
		GPUArithmetic::colConst<ArithmeticOperations::mod>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
		GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			if (inputDataBConst != 0)
			{
				ASSERT_EQ(outputData[i], inputDataA[i] % inputDataBConst);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////

	// Free buffers in GPU memory:
	GPUMemory::free(inputBufferA);
	GPUMemory::free(outputBuffer);
}

template<>
void testColConstArithmetic<float>()
{
	// CPU data:
	std::unique_ptr<float[]> inputDataA = std::make_unique<float[]>(DATA_ELEMENT_COUNT);
	float inputDataBConst;
	std::unique_ptr<float[]> outputData = std::make_unique<float[]>(DATA_ELEMENT_COUNT);

	// Fill input data buffers:
	std::default_random_engine generator;
	if (std::is_integral<float>::value)
	{
		std::uniform_int_distribution<int32_t> distributionInt(-1024, 1024);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionInt(generator);
		}
		inputDataBConst = distributionInt(generator);
	}
	else
	{
		std::uniform_real_distribution<float> distributionFloat(-1024.0f, 1024.0f);
		for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
		{
			inputDataA[i] = distributionFloat(generator);
		}
		inputDataBConst = distributionFloat(generator);
	}


	// Create CUDA buffers:
	float *inputBufferA;
	float *outputBuffer;

	// Alloc buffers in GPU memory:
	GPUMemory::alloc(&inputBufferA, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&outputBuffer, DATA_ELEMENT_COUNT);

	// Copy the contents of the buffers to the GPU
	GPUMemory::copyHostToDevice(inputBufferA, inputDataA.get(), DATA_ELEMENT_COUNT);

	//////////////////////////////////////////////////////////////////////////////////////
	// Run kernels, copy back values and compare them

	// Add
	GPUArithmetic::colConst<ArithmeticOperations::add>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] + inputDataBConst);
	}

	// Sub
	GPUArithmetic::colConst<ArithmeticOperations::sub>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] - inputDataBConst);
	}

	// Mul
	GPUArithmetic::colConst<ArithmeticOperations::mul>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		ASSERT_FLOAT_EQ(outputData[i], inputDataA[i] * inputDataBConst);
	}

	// Floor div
	GPUArithmetic::colConst<ArithmeticOperations::floorDiv>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);

	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (inputDataBConst != 0)
		{
			ASSERT_FLOAT_EQ(outputData[i], static_cast<float>(floor(inputDataA[i] / inputDataBConst)));
		}
	}

	// Div
	GPUArithmetic::colConst<ArithmeticOperations::div>(outputBuffer, inputBufferA, inputDataBConst, DATA_ELEMENT_COUNT);
	GPUMemory::copyDeviceToHost(outputData.get(), outputBuffer, DATA_ELEMENT_COUNT);
	for (int i = 0; i < DATA_ELEMENT_COUNT; i++)
	{
		if (inputDataBConst != 0)
		{
			ASSERT_FLOAT_EQ(outputData[i], static_cast<float>(inputDataA[i] / inputDataBConst));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////

	// Free buffers in GPU memory:
	GPUMemory::free(inputBufferA);
	GPUMemory::free(outputBuffer);
}

TEST(GPUArithmeticTests, ArithmeticsColConst)
{
	// Initialize CUDA context:
	Context::getInstance();

	testColConstArithmetic <int32_t>();
	testColConstArithmetic <int64_t>();
	testColConstArithmetic <float>();
}