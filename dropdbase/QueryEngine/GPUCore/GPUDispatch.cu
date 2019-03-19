#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../Context.h"
#include "../../DataType.h"
#include "GPUFilter.cuh"
#include "GPULogic.cuh"
#include "GPUDispatch.cuh"
#include "MaybeDeref.cuh"
#include "GpuMemory.cuh"

__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, void** symbols, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
		GPUStack<2048> gpuStack;
        int8_t registers[3];
		for (int32_t j = 0; j < opCodesCount; j++)
		{
            opCodes[i].fun_ptr(opCodes[i], i, gpuStack, registers, symbols);
		}
        outMask[i] = registers[0]; 
    }
}

__device__ void containsFunction(GPUOpCode opCode, int32_t offset, GPUStack<2048>& gpuStack, int8_t* registers, void** symbols)
{
	GPUMemory::GPUPolygon p;
	p.pointCount = gpuStack.pop<int32_t*>();
	p.pointIdx = gpuStack.pop<int32_t*>();
	p.polyCount = gpuStack.pop<int32_t*>();
	p.polyIdx = gpuStack.pop<int32_t*>();
	p.polyPoints = gpuStack.pop<NativeGeoPoint*>();
	//Zavolaj Contains,urob z neho device funkiu
}