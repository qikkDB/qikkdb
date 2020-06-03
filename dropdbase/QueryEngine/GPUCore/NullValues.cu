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
    return (size + sizeof(nullmask_t) * 8 - static_cast<nullmask_t>(1U)) / (sizeof(nullmask_t) * 8);
}

__device__ __host__ size_t NullValues::GetNullBitMaskSizeInBytes(const size_t size)
{
    return GetNullBitMaskSize(size) * sizeof(nullmask_t);
}

__device__ __host__ void
NullValues::SetBitInBitMask(nullmask_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx, const int8_t newBit)
{
    if (newBit)
    {
        bitMask[bitMaskIdx] |= (static_cast<nullmask_t>(1U) << shiftMaskIdx);
    }
    else
    {
        bitMask[bitMaskIdx] &= ~(static_cast<nullmask_t>(1U) << shiftMaskIdx);
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
    return (bitMask[bitMaskIdx] >> shiftMaskIdx) & static_cast<nullmask_t>(1U);
}

__device__ __host__ int8_t NullValues::GetConcreteBitFromBitmask(const nullmask_t* bitMask, const int32_t index)
{
    const int32_t bitMaskIdx = GetBitMaskIdx(index);
    const int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    return GetConcreteBitFromBitmask(bitMask, bitMaskIdx, shiftMaskIdx);
}

__device__ __host__ nullmask_t NullValues::GetPartOfBitmaskByte(const nullmask_t* bitMask,
                                                             const int32_t shiftMaskIdx,
                                                             const int32_t bitMaskIdx)
{
    return ((static_cast<nullmask_t>(1U) << (shiftMaskIdx + static_cast<nullmask_t>(1U))) -
            static_cast<nullmask_t>(1U)) &
           bitMask[bitMaskIdx];
}
