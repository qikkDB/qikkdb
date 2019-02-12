#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"

template<typename T, typename U>
__global__ void kernel_convert_buffer(T *outData, U *inData, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outData[i] = inData[i];
	}
}

class GPUTypeWidthManip {
public:
	template<typename T, typename U>
	static void convertBuffer(T *outData, U *inData, int32_t dataElementCount)
	{
		Context& context = Context::getInstance();

		kernel_convert_buffer << < context.calcGridDim(dataElementCount), context.getBlockDim() >> >
			(outData, inData, dataElementCount);

		context.getLastError().setCudaError(cudaGetLastError());
	}
};

