#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "MaybeDeref.cuh"

struct GPUOpCode
{
    int8_t(*fun_ptr)(GPUOpCode);
    void* dataLeft;
    void* dataRight;
    int32_t regIdx;
};

template <typename OP, typename L, typename R>
int8_t filterFunction(GPUOpCode opCode, int32_t offset)
{
    L* left = reinterpret_cast<L*>(opCode.dataLeft);
    R* left = reinterpret_cast<R*>(opCode.dataRight);
    return OP{}.template operator() <typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type> (maybe_deref(left, offset), maybe_deref(right, offset));
}


__global__ void kernel_filter(int8_t* outMask, GPUOpCode* opCodes, int32_t opCodesCount, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int8_t registers[3];
		for (int32_t j = 0; j < opCodesCount; j++)
		{
            registers[opCodes[i].regIdx] = opCodes[i].fun_ptr(opCodes[i]);
		}
        outMask[i] = registers[opCodes[opCodesCount - 1].regIdx]; 
    }
}