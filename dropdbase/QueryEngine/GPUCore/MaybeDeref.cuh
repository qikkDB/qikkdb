#pragma once

/// A hack used for column/constant operations to spare code
template <class T>
__device__ constexpr T maybe_deref(T* ptr, int i)
{
    return ptr[i];
}

/// A hack used for column/constant operations to spare code
template <class T>
__device__ constexpr T maybe_deref(T val, int i)
{
    return val;
}