#pragma once

class NullConstants
{
public:
    template <typename T>
    __device__ __host__ static constexpr T GetNullConstant()
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
};
