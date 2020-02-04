#include "NullValues.h"

int32_t NullValues::GetBitMaskIdx(const int32_t idx)
{
    return (idx / (sizeof(char) * 8));
}

int32_t NullValues::GetShiftMaskIdx(const int32_t idx)
{
    return (idx % (sizeof(char) * 8));
}

size_t NullValues::GetNullBitMaskSize(const size_t size)
{
    return (size + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
}

void NullValues::SetBitInBitMask(int8_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx, const int8_t newBit)
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

void NullValues::SetBitInBitMask(int8_t* bitMask, const int32_t index, const int8_t newBit)
{
    int32_t bitMaskIdx = GetBitMaskIdx(index);
    int32_t shiftMaskIdx = GetShiftMaskIdx(index);

    SetBitInBitMask(bitMask, bitMaskIdx, shiftMaskIdx, newBit);
}

int8_t NullValues::GetConcreteBitFromBitmask(int8_t* bitMask, const int32_t bitMaskIdx, const int32_t shiftMaskIdx)
{
    return (bitMask[bitMaskIdx] >> shiftMaskIdx) & 1;
}
