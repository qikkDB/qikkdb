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
    SetBitInBitMask(int64_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx, int8_t newBit);
    static void SetBitInBitMask(int64_t* bitMask, int32_t index, int8_t newBit);
    static int8_t GetConcreteBitFromBitmask(const int64_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx);
    static int8_t GetConcreteBitFromBitmask(const int64_t* bitMask, int32_t index);

	/// <summary>
    /// Get right part of bitmask Byte
    /// </summary>
    /// <param name="bitMask">bitmask where we are splitting byte</param>
    /// <param name="shiftMaskIdx">define idx in byte, which determine part we want to get<param>
    /// <param name="bitMaskIdx">define which byte we are splitting<param>
    static int64_t GetPartOfBitmaskByte(const int64_t* bitMask, int32_t shiftMaskIdx, int32_t bitMaskIdx);
};