#pragma once
#include <cstdint>
#include <memory>

/// Null Mask Type
typedef uint64_t nullmask_t;

/// Null Mask Type for CUDA Atomic Operations
typedef std::conditional<sizeof(nullmask_t) == 8, unsigned long long int, nullmask_t>::type nullmask_cuda_t;

/// Type for uncommpressed arrays with null values (one value per number)
typedef uint8_t nullarray_t;

class NullValues
{
public:
    static __device__ __host__ int32_t GetBitMaskIdx(int32_t idx);
    static __device__ __host__ int32_t GetShiftMaskIdx(int32_t idx);
    static __device__ __host__ size_t GetNullBitMaskSize(size_t size);
    static __device__ __host__ size_t GetNullBitMaskSizeInBytes(size_t size);
    static __device__ __host__ void 
    SetBitInBitMask(nullmask_t* bitMask, int32_t bitMaskIdx, int32_t shiftMaskIdx, nullmask_t newBit);
    static __device__ __host__ void SetBitInBitMask(nullmask_t* bitMask, int32_t index, nullmask_t newBit);
    static __device__ __host__ nullmask_t GetConcreteBitFromBitmask(const nullmask_t* bitMask,
                                                                int32_t bitMaskIdx,
                                                                int32_t shiftMaskIdx);
    static __device__ __host__ nullmask_t GetConcreteBitFromBitmask(const nullmask_t* bitMask, int32_t index);

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