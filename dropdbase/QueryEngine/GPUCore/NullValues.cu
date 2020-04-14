#include "NullValues.cuh"

__device__ __host__ int32_t NullValues::GetBitMaskIdx(const int32_t idx)
{
    return (idx / (sizeof(int64_t) * 8));
}

__device__ __host__ int32_t NullValues::GetShiftMaskIdx(const int32_t idx)
{
    return (idx % (sizeof(int64_t) * 8));
}

__device__ __host__ size_t NullValues::GetNullBitMaskSize(const size_t size)
{
    uint64_t longOne = 1;
    return (size + sizeof(int64_t) * 8 - longOne) / (sizeof(int64_t) * 8);
}

__device__ __host__ void NullValues::SetBitInBitMask(int64_t* bitMask,
                                                    const int32_t bitMaskIdx,
                                                    const int32_t shiftMaskIdx,
                                                    const int8_t newBit)
{
    uint64_t longOne = 1;
    if (newBit)
    {
        bitMask[bitMaskIdx] |= (longOne << shiftMaskIdx);
    }
    else
    {
        bitMask[bitMaskIdx] &= ~(longOne << shiftMaskIdx);
	}
}

__device__ __host__ void NullValues::SetBitInBitMask(int64_t* bitMask, const int32_t index, const int8_t newBit)
{
    int32_t bitMaskIdx = GetBitMaskIdx(index);
    int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    SetBitInBitMask(bitMask, bitMaskIdx, shiftMaskIdx, newBit);
}

__device__ __host__ int8_t NullValues::GetConcreteBitFromBitmask(const int64_t* bitMask,
                                                                const int32_t bitMaskIdx,
                                                                const int32_t shiftMaskIdx)
{
    uint64_t longOne = 1;

    return (bitMask[bitMaskIdx] >> shiftMaskIdx) & longOne;
}

__device__ __host__ int8_t NullValues::GetConcreteBitFromBitmask(const int64_t* bitMask, const int32_t index)
{
    int32_t bitMaskIdx = GetBitMaskIdx(index);
    int32_t shiftMaskIdx = GetShiftMaskIdx(index);

	return GetConcreteBitFromBitmask(bitMask, bitMaskIdx, shiftMaskIdx);
}

__device__ __host__ int64_t NullValues::GetPartOfBitmaskByte(const int64_t* bitMask, const int32_t shiftMaskIdx, const int32_t bitMaskIdx)
{
    uint64_t longOne = 1;

    return ((longOne << (shiftMaskIdx + longOne)) - longOne) &
                       bitMask[bitMaskIdx];
}
