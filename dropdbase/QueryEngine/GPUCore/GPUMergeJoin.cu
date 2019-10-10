#include "GPUMergeJoin.cuh"

__global__ void kernel_label_input(int32_t *colBlockIndices, int32_t blockOffset, int32_t dataElementCount) 
{
	const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < dataElementCount; i += stride)
	{
		colBlockIndices[i] = blockOffset + i;
	}
}

__global__ void kernel_compress_join_indices(int32_t* colABlockJoinIndices,
                                             int32_t* colBBlockJoinIndices,
                                             int8_t* joinPredicateMask,
                                             int32_t* joinPredicateMaskPSI,
                                             int32_t* mergeAIndices,
                                             int32_t* mergeBIndices,
                                             int32_t* colABlockIndices,
                                             int32_t* colBBlockIndices,
                                             int32_t diagonalCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < diagonalCount; i += stride)
    {
        if (joinPredicateMask[i])
        {
            int32_t joinPairIdx = (i == 0) ? 0 : joinPredicateMaskPSI[i - 1];

            colABlockJoinIndices[joinPairIdx] = colABlockIndices[mergeAIndices[i]];
            colBBlockJoinIndices[joinPairIdx] = colBBlockIndices[mergeBIndices[i]];
        }
    }
}