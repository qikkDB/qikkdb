#pragma once

/// How many times should be values and occurrences arrays duplicated
/// (for speedup atomic operations)
constexpr int32_t GB_VALUE_BUFFER_DEFAULT_MULTIPLIER = 256; // should be multiple of 32
constexpr size_t GB_BUFFER_SIZE_MAX = 1 << 30; // should be less than INT_MAX

/// Interface for abstracting GPUGroupBy instances
class IGroupBy
{
protected:
    IGroupBy()
    {
    }

public:
    virtual ~IGroupBy()
    {
    }
};
