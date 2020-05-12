#include "GPUOrderBy.cuh"
#include "GPUStringUnary.cuh"
#include "GPUReconstruct.cuh"

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

__global__ void kernel_reorder_polyCount_col(int32_t* outPolyCount,
										     int32_t* inIndices, 
										     GPUMemory::GPUPolygon inPolygon,
										     int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t polyIdx = inPolygon.PolyIdxAt(inIndices[i]);
        int32_t polyCount = inPolygon.PolyCountAt(inIndices[i]);

        outPolyCount[i] = polyCount;
    }

}

__global__ void kernel_reorder_pointCount_col(int32_t* outPointCount,
                                              GPUMemory::GPUPolygon outPolygon,
                                              int32_t* inIndices, 
                                              GPUMemory::GPUPolygon inPolygon,
                                              int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t polyIdx = inPolygon.PolyIdxAt(inIndices[i]);
        int32_t polyCount = inPolygon.PolyCountAt(inIndices[i]);

		int32_t outPolyIdx = outPolygon.PolyIdxAt(i);
        int32_t outPolyCount = outPolygon.PolyCountAt(i);

        for (int32_t p = polyIdx, op = outPolyIdx; p < (polyIdx + polyCount); p++, op++)
        {
            int32_t pointIdx = inPolygon.PointIdxAt(p);
            int32_t pointCount = inPolygon.PointCountAt(p);

			outPointCount[op] = pointCount;
        }
    }
}

__global__ void kernel_reorder_polyPoints_col(GPUMemory::GPUPolygon outPolygon,
                                              int32_t* inIndices, 
                                              GPUMemory::GPUPolygon inPolygon,
                                              int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t polyIdx = inPolygon.PolyIdxAt(inIndices[i]);
        int32_t polyCount = inPolygon.PolyCountAt(inIndices[i]);

		int32_t outPolyIdx = outPolygon.PolyIdxAt(i);
        int32_t outPolyCount = outPolygon.PolyCountAt(i);

        for (int32_t p = polyIdx, op = outPolyIdx; p < (polyIdx + polyCount); p++, op++)
        {
            int32_t pointIdx = inPolygon.PointIdxAt(p);
            int32_t pointCount = inPolygon.PointCountAt(p);

			int32_t outPointIdx = outPolygon.PointIdxAt(op);
            int32_t outPointCount = outPolygon.PointCountAt(op);

            for (int32_t point = pointIdx, oPoint = outPointIdx; point < (pointIdx + pointCount); point++, oPoint++)
            {
                outPolygon.polyPoints[oPoint] = inPolygon.polyPoints[point];
            }
        }
    }
}

__global__ void kernel_reorder_point_counts_by_poly_idx_lenghts(int32_t* outPointLengths,
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
        const int32_t outPointIdx = outPolygonIndices[i];
        const int32_t inPointIdx = inPolygonIndices[inOrderIndices[i]];
        for (int32_t j = 0; j < outPolygonLengths[i]; j++)
        {
            outPointLengths[outPointIdx + j] = inPointLenghts[inPointIdx + j];
        }
    }
}

// Reorder a null column by a given index column
__global__ void kernel_reorder_null_values_by_idx(nullmask_t* outNullBitMask,
                                                  int32_t* inIndices,
                                                  nullmask_t* inNullBitMask,
                                                  int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        nullmask_t nullBit = NullValues::GetConcreteBitFromBitmask(inNullBitMask, inIndices[i]);
        nullBit <<= NullValues::GetShiftMaskIdx(i);
        atomicOr(reinterpret_cast<unsigned long long int*>(outNullBitMask) + NullValues::GetBitMaskIdx(i), nullBit);
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

void GPUOrderBy::ReOrderNullValuesByIdx(nullmask_t* outNullBitMask, int32_t* indices, nullmask_t* inNullBitMask, int32_t dataElementCount)
{
    if (inNullBitMask != nullptr)
    {
        // Zero the out mask
        GPUMemory::fillArray(outNullBitMask, static_cast<nullmask_t>(0), dataElementCount);

        // Reorder the bits
        kernel_reorder_null_values_by_idx<<<Context::getInstance().calcGridDim(dataElementCount),
                                            Context::getInstance().getBlockDim()>>>(
            reinterpret_cast<nullmask_t*>(outNullBitMask), indices, inNullBitMask, dataElementCount);
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
		// Reorder polygons
        cuda_ptr<int32_t> polyCount(dataElementCount);
        GPUMemory::alloc(&outCol.polyIdx, dataElementCount);

        kernel_reorder_polyCount_col<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            polyCount.get(), inIndices, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUReconstruct::PrefixSum(outCol.polyIdx, polyCount.get(), dataElementCount);

		// Reorder sub polygons
        int32_t pointCountSize;
        GPUMemory::copyDeviceToHost(&pointCountSize, inCol.polyIdx + dataElementCount - 1, 1);

		cuda_ptr<int32_t> pointCount(pointCountSize);
        GPUMemory::alloc(&outCol.pointIdx, pointCountSize);

		kernel_reorder_pointCount_col<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            pointCount.get(), outCol, inIndices, inCol, dataElementCount);
        CheckCudaError(cudaGetLastError());

        GPUReconstruct::PrefixSum(outCol.pointIdx, pointCount.get(), pointCountSize);

		// Reorder points
        int32_t polyPointSize;
        GPUMemory::copyDeviceToHost(&polyPointSize, inCol.pointIdx + pointCountSize - 1, 1);

		GPUMemory::alloc(&outCol.polyPoints, polyPointSize);

        kernel_reorder_polyPoints_col<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outCol, inIndices, inCol, dataElementCount);
    }
    else
    {
        outCol.polyPoints = nullptr;
        outCol.pointIdx = nullptr;
        outCol.polyIdx = nullptr;
    }
}
