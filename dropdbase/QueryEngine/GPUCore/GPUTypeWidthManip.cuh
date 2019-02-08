#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"
#include "GPUConstants.cuh"

template<typename T, typename U>
__global__ void kernel_convert_buffer(T *outData, U *inData, int32_t dataElementCount)
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;
	const int32_t loopIterations = (dataElementCount + stride - 1 - idx) / stride;
	const int32_t alignedLoopIterations = loopIterations - (loopIterations % UNROLL_FACTOR);
	const int32_t alignedDataElementCount = alignedLoopIterations * stride + idx;

	//unroll from idx to alignedDataElementCount
	#pragma unroll UNROLL_FACTOR
	for (int32_t i = idx; i < alignedDataElementCount; i += stride)
	{
		outData[i] = inData[i];
	}
	//continue classic way from alignedDataElementCount to full dataElementCount
	for (int32_t i = alignedDataElementCount; i < dataElementCount; i += stride)
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

