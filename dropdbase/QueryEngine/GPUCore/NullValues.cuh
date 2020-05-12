#pragma once
#include <cstdint>
#include <memory>

typedef uint32_t nullmask_t;

class NullValues
{
public:
    static __device__ __host__ int32_t GetBitMaskIdx(int32_t idx);
    static __device__ __host__ int32_t GetShiftMaskIdx(int32_t idx);
    static __device__ __host__ size_t GetNullBitMaskSize(size_t size);
    static __device__ __host__ void 
    SetBitInBitMask(nullmask_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx, int8_t newBit);
    static __device__ __host__ void SetBitInBitMask(nullmask_t* bitMask, int32_t index, int8_t newBit);
    static __device__ __host__ int8_t GetConcreteBitFromBitmask(const nullmask_t* bitMask,
                                                                int32_t bitMaskIdx,
                                                                int32_t shiftMaskIdx);
    static __device__ __host__ int8_t GetConcreteBitFromBitmask(const nullmask_t* bitMask, int32_t index);

	/// <summary>
    /// Get right part of bitmask Byte
    /// </summary>
    /// <param name="bitMask">bitmask where we are splitting byte</param>
    /// <param name="shiftMaskIdx">define idx in byte, which determine part we want to get<param>
    /// <param name="bitMaskIdx">define which byte we are splitting<param>
    static __device__ __host__ nullmask_t GetPartOfBitmaskByte(const nullmask_t* bitMask,
                                                            int32_t shiftMaskIdx,
                                                            int32_t bitMaskIdx);
};