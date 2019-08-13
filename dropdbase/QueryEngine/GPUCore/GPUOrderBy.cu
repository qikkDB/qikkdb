#include "GPUOrderBy.cuh"
#include "GPUStringUnary.cuh"

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
                                            int64_t* outStringIndices,
                                            int32_t* outStringLengths,
                                            int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outColIdx = GetStringIndex(outStringIndices, i);
        int32_t inColIdx = GetStringIndex(inCol.stringIndices, inIndices[i]);
        for (int32_t j = 0; j < outStringLengths[i]; j++)
        {
            outCol.allChars[outColIdx + j] = inCol.allChars[inColIdx + j];
        }
        outCol.stringIndices[i] = outStringIndices[i];
    }
}

__global__ void kernel_reorder_points_by_idx(GPUMemory::GPUPolygon outCol,
                                             int32_t* inIndices,
                                             GPUMemory::GPUPolygon inCol,
                                             int32_t* outComplexPolygonIndices,
                                             int32_t* outComplexPolygonLengths,
                                             int32_t* outPolygonIndices,
                                             int32_t* outPolygonLengths,
                                             int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outPolygonIdx = GetStringIndex(outComplexPolygonIndices, i);
        int32_t inPolygonIdx = GetStringIndex(inCol.polyIdx, inIndices[i]);
        for (int32_t j = 0; j < outComplexPolygonLengths[i]; j++)
        {
            int32_t outPointIdx = GetStringIndex(outPolygonIndices, outPolygonIdx);
            for (int32_t k = 0; k < outPolygonLengths[outPolygonIdx]; k++)
            {

            }
            outCol.pointIdx[outPolygonIdx] = inCol.pointIdx[inPolygonIdx + j];
            outCol.pointCount[outPolygonIdx] = inCol.pointCount[inPolygonIdx + j];
        }
        outCol.polyIdx[i] = outComplexPolygonIndices[i];
        outCol.polyCount[i] = outComplexPolygonLengths[i];
    }
}

__global__ void kernel_reorder_by_idx_and_lenghts(int32_t* outCol,
                                                  int32_t* inIndices,
                                                  int32_t* inCol,
                                                  int32_t* outIndices,
                                                  int32_t* outLengths,
                                                  int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outColIdx = GetStringIndex(outIndices, i);
        int32_t inColIdx = GetStringIndex(inCol, inIndices[i]);
        for (int32_t j = 0; j < outLengths[i]; j++)
        {
            outCol[outColIdx + j] = inCol[inColIdx + j];
        }
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

    if (dataElementCount > 0)
    {

        cuda_ptr<int32_t> inStringLengths(dataElementCount);
        kernel_lengths_from_indices<int32_t, int64_t>
            <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(inStringLengths.get(),
                                                                               inCol.stringIndices,
                                                                               dataElementCount);
        cuda_ptr<int32_t> outStringLengths(dataElementCount);
        kernel_reorder_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outStringLengths.get(), inIndices, inStringLengths.get(), dataElementCount);
        cuda_ptr<int64_t> outStringIndices(dataElementCount);
        GPUReconstruct::PrefixSum(outStringIndices.get(), outStringLengths.get(), dataElementCount);
        GPUMemory::alloc(&outCol.stringIndices, dataElementCount);

        int64_t totalCharCount;
        GPUMemory::copyDeviceToHost(&totalCharCount, &inCol.stringIndices[dataElementCount - 1], 1);
        GPUMemory::alloc(&outCol.allChars, totalCharCount);

        kernel_reorder_chars_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outCol, inIndices, inCol, outStringIndices.get(), outStringLengths.get(), dataElementCount);
    }
    else
    {
        outCol.stringIndices = nullptr;
        outCol.allChars = nullptr;
    }
}

void GPUOrderBy::ReOrderPolygonByIdx(GPUMemory::GPUPolygon& outCol,
                                     int32_t* inIndices,
                                     GPUMemory::GPUPolygon inCol,
                                     int32_t dataElementCount)
{
    Context& context = Context::getInstance();

    if (dataElementCount > 0)
    {

        cuda_ptr<int32_t> inComplexPolygonLengths(dataElementCount);
        kernel_lengths_from_indices<int32_t, int32_t>
            <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
                inComplexPolygonLengths.get(), inCol.polyIdx, dataElementCount);

        cuda_ptr<int32_t> outComplexPolygonLengths(dataElementCount);
        kernel_reorder_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outComplexPolygonLengths.get(), inIndices, inComplexPolygonLengths.get(), dataElementCount);

        cuda_ptr<int32_t> outComplexPolygonIndices(dataElementCount);
        GPUReconstruct::PrefixSum(outComplexPolygonIndices.get(), outComplexPolygonLengths.get(), dataElementCount);

        int32_t totalPolygonCount;
        GPUMemory::copyDeviceToHost(&totalPolygonCount, &inCol.polyIdx[dataElementCount - 1], 1);

        cuda_ptr<int32_t> inPolygonLengths(totalPolygonCount);
        kernel_lengths_from_indices<int32_t, int32_t>
            <<<context.calcGridDim(totalPolygonCount), context.getBlockDim()>>>(inPolygonLengths.get(),
                                                                                inCol.pointIdx,
                                                                                totalPolygonCount);

        cuda_ptr<int32_t> outPolygonLengths(totalPolygonCount);
        kernel_reorder_by_idx_and_lenghts<<<context.calcGridDim(totalPolygonCount), context.getBlockDim()>>>(
            outPolygonLengths.get(), inIndices, inPolygonLengths.get(),
            outComplexPolygonIndices.get(), outComplexPolygonLengths.get(), totalPolygonCount);

        cuda_ptr<int32_t> outPolygonIndices(totalPolygonCount);
        GPUReconstruct::PrefixSum(outPolygonIndices.get(), outPolygonLengths.get(), totalPolygonCount);

        int32_t totalPointCount;
        GPUMemory::copyDeviceToHost(&totalPointCount, &outPolygonIndices.get()[totalPolygonCount - 1], 1);

        kernel_reorder_points_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outCol, inIndices, inCol, outComplexPolygonIndices.get(), outComplexPolygonLengths.get(),
            outPolygonIndices.get(), outPolygonLengths.get(), dataElementCount);
    }
    else
    {
        outCol.polyPoints = nullptr;
        outCol.pointIdx = nullptr;
        outCol.pointCount = nullptr;
        outCol.polyIdx = nullptr;
        outCol.polyCount = nullptr;
    }
}
