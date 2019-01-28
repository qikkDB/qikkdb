#ifndef GPU_GROUP_BY_CUH
#define GPU_GROUP_BY_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Context.h"

template<typename AGG, typename K, typename V>
__global__ void group_by_kernel(K *outKeys, V *outValues, int32_t outDataElementCount, K *inKeys, V *inValues, int32_t dataElementCount)
{
	// DEMO mock
	outKeys[0] = AGG{}(5, 4);
}

class GPUGroupBy
{
public:

	// Generic agg function functors
	struct min { template<typename T> __device__ T operator()(T a, T b) const { return a < b ? a : b; } };
	struct max { template<typename T> __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
	struct sum { template<typename T> __device__ T operator()(T a, T b) const { return a + b; } };
	//struct avg { void operator()() const {} };
	//struct cnt { void operator()() const {} };

	template<typename AGG, typename K, typename V>
	static void groupBy(K *outKeys, V *outValues, int32_t outDataElementCount, K *inKeys, V *inValues, int32_t dataElementCount)
	{
		// DEMO mock
		group_by_kernel <AGG> << <1, 1 >> >
			(outKeys, outValues, outDataElementCount, inKeys, inValues, dataElementCount);
		cudaDeviceSynchronize();

		Context::getInstance().getLastError().setCudaError(cudaGetLastError());
	}
};

#endif