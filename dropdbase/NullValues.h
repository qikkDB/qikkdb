#pragma once
#include <cstdint>
#include <memory>

class NullValues
{
public:
    static int32_t GetBitMaskIdx(int32_t idx);
    static int32_t GetShiftMaskIdx(int32_t idx);
    static size_t GetNullBitMaskSize(size_t size);
    static void
    SetBitInBitMask(int8_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx, int8_t newBit);
    static void SetBitInBitMask(int8_t* bitMask, int32_t index, int8_t newBit);
    static int8_t GetConcreteBitFromBitmask(int8_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx);
};