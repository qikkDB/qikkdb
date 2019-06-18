#pragma once

#include <cuda_runtime.h>

/// This function returns constant of type T which is used as null
/// (e.g. result of division by 0 should be this constant)
/// <returns>"null" constant</returns>
template <typename T>
__device__ __host__ constexpr T GetNullConstant()
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                    "Unsupported data type (in function GetNullConstant)");

    if (std::is_integral<T>::value)
    {
        return std::numeric_limits<T>::min();
    }
    else if (std::is_floating_point<T>::value)
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    else
    {
        return T{};
    }
}
