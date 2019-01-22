#ifndef GPU_TYPE_WIDTH_MANIP_H
#define GPU_TYPE_WIDTH_MANIP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.cuh"
#include "../InterfaceCore/ITypeWidthManip.h"

template<typename T, typename U>
__global__ void kernel_convert_buffer(T *outData, U *inData, int32_t dataElementCount)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		outData[i] = inData[i];
	}
}

class GPUTypeWidthManip : public ITypeWidthManip {
private:

public:
	template<typename T, typename U>
	void ConvertBuffer(T *outData, U *inData, int32_t dataElementCount)
	{
		
		kernel_convert_buffer << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outData, inData, dataElementCount);
	}
};

#endif

