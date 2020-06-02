#include "GPUStringUnary.cuh"

__device__ int64_t GetStringIndex(int64_t* indices, const int64_t i)
{
    return (i == 0) ? 0 : indices[i - 1];
}

__device__ int32_t GetStringLength(int64_t* indices, const int64_t i)
{
    return static_cast<int32_t>(indices[i] - GetStringIndex(indices, i));
}

__global__ void kernel_reverse_string(GPUMemory::GPUString outCol, GPUMemory::GPUString inCol, int32_t stringCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < stringCount; i += stride)
    {
        const int64_t firstCharIndex = GetStringIndex(inCol.stringIndices, i);
        const int64_t length = inCol.stringIndices[i] - firstCharIndex;
        const int64_t lastCharIndex = inCol.stringIndices[i] - 1;

        for (int32_t j = 0; j < length; j++)
        {
            outCol.allChars[firstCharIndex + j] = inCol.allChars[lastCharIndex - j];
        }
    }
}

template <>
void StringUnaryOpHierarchy::fixed::CallKernel<StringUnaryOpHierarchy::FixedLength::reverse>(
    GPUMemory::GPUString outCol,
    GPUMemory::GPUString input,
    int32_t stringCount,
    int64_t totalCharCount)
{
    Context& context = Context::getInstance();
    kernel_reverse_string<<<context.calcGridDim(stringCount), context.getBlockDim()>>>(outCol, input, stringCount);
}
