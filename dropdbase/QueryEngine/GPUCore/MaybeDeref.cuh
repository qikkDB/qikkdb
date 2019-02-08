#pragma once

template<class T>
__device__ constexpr T maybe_deref(T* ptr, int i) { return ptr[i]; }

template<class T>
__device__ constexpr T maybe_deref(T val, int i) { return val; }