#include "GPUOrderBy.cuh"

// Fill the index buffers with default indices
__global__ void kernel_fill_indices(int32_t* indices, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        indices[i] = i;
    }
}

__global__ void kernel_reorder_chars_by_idx(GPUMemory::GPUString outCol,
                                            int32_t* inIndices,
                                            GPUMemory::GPUString inCol,
                                            int32_t* outStringIndices,
                                            int32_t* outStringLenghts,
                                            int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outColIdx = (i == 0) ? 0 : outStringIndices[i - 1];
        int32_t inColIdx = (inIndices[i] == 0) ? 0 : inCol.stringIndices[inIndices[i] - 1];
        for (int32_t j = 0; j < outStringLenghts[i]; j++)
        {
            outCol.allChars[outColIdx + j] = inCol.allChars[inColIdx + j];
        }
        outCol.stringIndices[i] = outStringIndices[i];
    }
}

// Reorder a null column by a given index column
__global__ void
kernel_reorder_null_values_by_idx(int32_t* outNullBitMask, int32_t* inIndices, int8_t* inNullBitMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t nullBit =
            (inNullBitMask[inIndices[i] / (sizeof(int8_t) * 8)] >> (inIndices[i] % (sizeof(int8_t) * 8))) & 1;
        nullBit <<= (i % (sizeof(int32_t) * 8));
        atomicOr(outNullBitMask + (i / (sizeof(int32_t) * 8)), nullBit);
    }
}

GPUOrderBy::GPUOrderBy(int32_t dataElementCount)
{
    GPUMemory::alloc(&indices1, dataElementCount);
    GPUMemory::alloc(&indices2, dataElementCount);

    // Initialize the index buffer
    kernel_fill_indices<<<Context::getInstance().calcGridDim(dataElementCount),
                          Context::getInstance().getBlockDim()>>>(indices1, dataElementCount);
}

GPUOrderBy::~GPUOrderBy()
{
    GPUMemory::free(indices1);
    GPUMemory::free(indices2);
}

void GPUOrderBy::ReOrderNullValuesByIdx(int8_t* outNullBitMask, int32_t* indices, int8_t* inNullBitMask, int32_t dataElementCount)
{
    if (inNullBitMask != nullptr)
    {
        // Zero the out mask
        GPUMemory::fillArray(outNullBitMask, static_cast<int8_t>(0), dataElementCount);

        // Reorder the bits
        kernel_reorder_null_values_by_idx<<<Context::getInstance().calcGridDim(dataElementCount),
                                            Context::getInstance().getBlockDim()>>>(
            reinterpret_cast<int32_t*>(outNullBitMask), indices, inNullBitMask, dataElementCount);
    }
}

void GPUOrderBy::ReOrderStringByIdx(GPUMemory::GPUString& outCol, int32_t* inIndices, GPUMemory::GPUString inCol, int32_t dataElementCount)
{
    Context& context = Context::getInstance();

    cuda_ptr<int32_t> inStringLengths(dataElementCount);
    kernel_lengths_from_indices<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
        inStringLengths.get(), inCol.stringIndices, dataElementCount);

    cuda_ptr<int32_t> outStringLengths(dataElementCount);
    kernel_reorder_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
        outStringLengths.get(), inIndices, inStringLengths.get(), dataElementCount);

    cuda_ptr<int32_t> outStringIndices(dataElementCount);
    GPUReconstruct::PrefixSumExclusive<int32_t>(outStringIndices.get(), outStringLengths.get(), dataElementCount);

	GPUMemory::alloc(&outCol.stringIndices, dataElementCount);

    int64_t totalCharCount;
    GPUMemory::copyDeviceToHost(&totalCharCount, inCol.stringIndices + dataElementCount - 1, 1);
    GPUMemory::alloc(&outCol.stringIndices, totalCharCount);

    kernel_reorder_chars_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
        outCol, inIndices, inCol, outStringIndices.get(), outStringLengths.get(), dataElementCount);
}
