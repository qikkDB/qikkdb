#include "NullValues.h"
#include "NullValues.h"
#include "NullValues.h"

int32_t NullValues::GetBitMaskIdx(const int32_t idx)
{
    return (idx / (sizeof(int64_t) * 8));
}

int32_t NullValues::GetShiftMaskIdx(const int32_t idx)
{
    return (idx % (sizeof(int64_t) * 8));
}

size_t NullValues::GetNullBitMaskSize(const size_t size)
{
    return (size + sizeof(int64_t) * 8 - 1) / (sizeof(int64_t) * 8);
}

void NullValues::SetBitInBitMask(int64_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx, const int8_t newBit)
{
    if (newBit)
    {
        bitMask[bitMaskIdx] |= (newBit << shiftMaskIdx);
    }
    else
    {
        bitMask[bitMaskIdx] &= ~(1 << shiftMaskIdx);
	}
}

void NullValues::SetBitInBitMask(int64_t* bitMask, const int32_t index, const int8_t newBit)
{
    int32_t bitMaskIdx = GetBitMaskIdx(index);
    int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    SetBitInBitMask(bitMask, bitMaskIdx, shiftMaskIdx, newBit);
}

int8_t NullValues::GetConcreteBitFromBitmask(const int64_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx)
{
    return (bitMask[bitMaskIdx] >> shiftMaskIdx) & 1;
}

int8_t NullValues::GetConcreteBitFromBitmask(const int64_t* bitMask, int32_t index)
{
    int32_t bitMaskIdx = GetBitMaskIdx(index);
    int32_t shiftMaskIdx = GetShiftMaskIdx(index);

	return GetConcreteBitFromBitmask(bitMask, bitMaskIdx, shiftMaskIdx);
}

int64_t NullValues::GetPartOfBitmaskByte(const int64_t* bitMask, int32_t shiftMaskIdx, int32_t bitMaskIdx)
{
    return ((1 << (shiftMaskIdx + 1)) - 1) & bitMask[bitMaskIdx];
}
