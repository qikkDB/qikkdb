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

__device__ int64_t GetPolygonIndex(int32_t* indices, const int32_t i)
{
    return (i == 0) ? 0 : indices[i - 1];
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
                                             int32_t* outPolygonIndices,
                                             int32_t* outPolygonLengths,
                                             int32_t* outPointIndices,
                                             int32_t* outPointLengths,
                                             int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outPolygonIdx = GetPolygonIndex(outPolygonIndices, i);
        int32_t inPolygonIdx = GetPolygonIndex(inCol.polyIdx, inIndices[i]);
        for (int32_t j = 0; j < outPolygonLengths[i]; j++)
        {
            int32_t outPointIdx = GetPolygonIndex(outPointIndices, outPolygonIdx);
            int32_t inPointIdx = GetPolygonIndex(inCol.pointIdx, inPolygonIdx);
            for (int32_t k = 0; k < outPointLengths[outPolygonIdx]; k++)
            {
                outCol.polyPoints[outPointIdx + k] = inCol.polyPoints[inPointIdx + k];
            }
            outCol.pointIdx[outPolygonIdx + j] = outPointIndices[inPolygonIdx + j];
            outCol.pointCount[outPolygonIdx + j] = outPointLengths[inPolygonIdx + j];
        }
        outCol.polyIdx[i] = outPolygonIndices[i];
        outCol.polyCount[i] = outPolygonLengths[i];
    }
}

__global__ void kernel_reorder_poly_lengths_by_cp_idx_and_cp_lenghts(int32_t* outPointLengths,
                                                                     int32_t* inOrderIndices,
                                                                     int32_t* inPointLenghts,
                                                                     int32_t* inPolygonIndices,
                                                                     int32_t* outPolygonIndices,
                                                                     int32_t* outPolygonLengths,
                                                                     int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t outPointIdx = GetPolygonIndex(outPolygonIndices, i);
        int32_t inPointIdx = GetPolygonIndex(inPolygonIndices, inOrderIndices[i]);
        for (int32_t j = 0; j < outPolygonLengths[i]; j++)
        {
            outPointLengths[outPointIdx + j] = inPointLenghts[inPointIdx + j];
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

        cuda_ptr<int32_t> inPolygonLengths(dataElementCount);
        kernel_lengths_from_indices<int32_t, int32_t>
            <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(inPolygonLengths.get(),
                                                                               inCol.polyIdx, dataElementCount);

        cuda_ptr<int32_t> outPolygonLengths(dataElementCount);
        kernel_reorder_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outPolygonLengths.get(), inIndices, inPolygonLengths.get(), dataElementCount);

        cuda_ptr<int32_t> outPolygonIndices(dataElementCount);
        GPUReconstruct::PrefixSum(outPolygonIndices.get(), outPolygonLengths.get(), dataElementCount);

        int32_t totalPolygonCount;
        GPUMemory::copyDeviceToHost(&totalPolygonCount, &inCol.polyIdx[dataElementCount - 1], 1);

        cuda_ptr<int32_t> inPointLengths(totalPolygonCount);
        kernel_lengths_from_indices<int32_t, int32_t>
            <<<context.calcGridDim(totalPolygonCount), context.getBlockDim()>>>(inPointLengths.get(),
                                                                                inCol.pointIdx,
                                                                                totalPolygonCount);

        cuda_ptr<int32_t> outPointLengths(totalPolygonCount);
        kernel_reorder_poly_lengths_by_cp_idx_and_cp_lenghts<<<context.calcGridDim(dataElementCount),
                                                               context.getBlockDim()>>>(
            outPointLengths.get(), inIndices, inPointLengths.get(), inCol.polyIdx,
            outPolygonIndices.get(), outPolygonLengths.get(), dataElementCount);

        cuda_ptr<int32_t> outPointIndices(totalPolygonCount);
        GPUReconstruct::PrefixSum(outPointIndices.get(), outPointLengths.get(), totalPolygonCount);

        int32_t totalPointCount;
        GPUMemory::copyDeviceToHost(&totalPointCount, &outPointIndices.get()[totalPolygonCount - 1], 1);

        GPUMemory::alloc(&outCol.polyCount, dataElementCount);
        GPUMemory::alloc(&outCol.polyIdx, dataElementCount);
        GPUMemory::alloc(&outCol.pointCount, totalPolygonCount);
        GPUMemory::alloc(&outCol.pointIdx, totalPolygonCount);
        GPUMemory::alloc(&outCol.polyPoints, totalPointCount);

        kernel_reorder_points_by_idx<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outCol, inIndices, inCol, outPolygonIndices.get(), outPolygonLengths.get(),
            outPointIndices.get(), outPointLengths.get(), dataElementCount);
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
