#ifndef GPU_FILTER_CUH
#define GPU_FILTER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.cuh"

//////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Kernel for comparing values from two columns - operator greater than (>)
/// </summary>
/// <param name="outMask">block of the result data</param>
/// <param name="ACol">block of the left input operands</param>
/// <param name="BCol">block of the right input operands</param>
/// <param name="dataType">Input data type</param>
/// <param name="dataElementCount">the size of the input blocks in bytes</param>
/// <returns>if operation was successful (GPU_EXTENSION_SUCCESS or GPU_EXTENSION_ERROR)</returns>
template<typename T, typename U>
__global__ void kernel_gt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < dataElementCount; i += stride)
	{
		outMask[i] = ACol[i] > BCol[i];
	}
}

class GPUFilter {
public:
	// Operator >
	template<typename T, typename U>
	void gt(int8_t *outMask, T *ACol, U *BCol, int32_t dataElementCount) const {
		kernel_gt<T, U> << < Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim() >> >
			(outMask, ACol, BCol, dataElementCount);
		cudaDeviceSynchronize();
		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}

};

#endif
