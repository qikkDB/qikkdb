#include "NullValues.cuh"

__device__ __host__ int32_t NullValues::GetBitMaskIdx(const int32_t idx)
{
    return (idx / (sizeof(nullmask_t) * 8));
}

__device__ __host__ int32_t NullValues::GetShiftMaskIdx(const int32_t idx)
{
    return (idx % (sizeof(nullmask_t) * 8));
}

__device__ __host__ size_t NullValues::GetNullBitMaskSize(const size_t size)
{
    return (size + sizeof(nullmask_t) * 8 - 1ULL) / (sizeof(nullmask_t) * 8);
}

__device__ __host__ void
NullValues::SetBitInBitMask(nullmask_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx, const int8_t newBit)
{
    if (newBit)
    {
        bitMask[bitMaskIdx] |= (1ULL << shiftMaskIdx);
    }
    else
    {
        bitMask[bitMaskIdx] &= ~(1ULL << shiftMaskIdx);
    }
}

__device__ __host__ void NullValues::SetBitInBitMask(nullmask_t* bitMask, const int32_t index, const int8_t newBit)
{
    const int32_t bitMaskIdx = GetBitMaskIdx(index);
    const int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    SetBitInBitMask(bitMask, bitMaskIdx, shiftMaskIdx, newBit);
}

__device__ __host__ int8_t NullValues::GetConcreteBitFromBitmask(const nullmask_t* bitMask,
                                                                 const int32_t bitMaskIdx,
                                                                 const int32_t shiftMaskIdx)
{
    return (bitMask[bitMaskIdx] >> shiftMaskIdx) & 1ULL;
}

__device__ __host__ int8_t NullValues::GetConcreteBitFromBitmask(const nullmask_t* bitMask, const int32_t index)
{
    const int32_t bitMaskIdx = GetBitMaskIdx(index);
    const int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    return GetConcreteBitFromBitmask(bitMask, bitMaskIdx, shiftMaskIdx);
}

__device__ __host__ int64_t NullValues::GetPartOfBitmaskByte(const nullmask_t* bitMask,
                                                             const int32_t shiftMaskIdx,
                                                             const int32_t bitMaskIdx)
{
    return ((1ULL << (shiftMaskIdx + 1ULL)) - 1ULL) & bitMask[bitMaskIdx];
}
