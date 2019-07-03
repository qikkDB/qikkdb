#pragma once
#include "macro.cuh"
//TODO: better distinguishing between signed/unsigned versions
template <typename T>
__device__ __host__ __forceinline__ T SETNPBITS( T *source, T value, const unsigned int num_bits, const unsigned int bit_start)
{
    T mask = BITMASK(T, num_bits);
    *source &= ~(mask<<bit_start); // clear space in source
    *source |= (value & mask) << bit_start; // set values
    return *source;
}

// GETNPBITS
__device__ __host__ __forceinline__ unsigned int GETNPBITS( int source, unsigned int num_bits, unsigned int bit_start)
{
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned int GETNPBITS( unsigned int source, unsigned int num_bits, unsigned int bit_start)
{
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( long source, unsigned int num_bits, unsigned int bit_start)
{
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( unsigned long source, unsigned int num_bits, unsigned int bit_start)
{
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long long GETNPBITS(long long source, unsigned int num_bits, unsigned int bit_start)
{
	return ((source >> bit_start) & LLNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long long GETNPBITS(unsigned long long source, unsigned int num_bits, unsigned int bit_start)
{
	return ((source >> bit_start) & LLNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS(float source, unsigned int num_bits, unsigned int bit_start)
{
	int source_int = *((int *)&source);
	return ((source_int >> bit_start) & LLNBITSTOMASK(num_bits));
}

//GETNBITS

__device__ __host__ __forceinline__ unsigned long GETNBITS( long source, unsigned int num_bits)
{
    return ((source) & LNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( unsigned long source, unsigned int num_bits)
{
    return ((source) & LNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( int source, unsigned int num_bits)
{
    return ((source) & NBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( unsigned int source, unsigned int num_bits)
{
    return ((source) & NBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long long GETNBITS(long long source, unsigned int num_bits)
{
	return ((source)& LLNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long long GETNBITS(unsigned long long source, unsigned int num_bits)
{
	return ((source)& LLNBITSTOMASK(num_bits));
}

__device__ __host__ __forceinline__ unsigned long GETNBITS(float source, unsigned int num_bits)
{
	int source_int = *((int *)&source);
	return ((source_int)& LNBITSTOMASK(num_bits));
}

//BITLEN

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned int word)
{
    unsigned int ret=0;
    while (word >>= 1)
      ret++;
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned long word)
{
    unsigned int ret=0;
    while (word >>= 1)
      ret++;
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned long long word)
{
	unsigned int ret = 0;
	while (word >>= 1)
		ret++;
	return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(int word)
{
	unsigned int uword = (unsigned int) word;
    unsigned int ret=0;
    while (uword >>= 1)
      ret++;
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(long word)
{
	unsigned long uword = (unsigned long)word;
    unsigned int ret=0;
    while (uword >>= 1)
      ret++;
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(long long word)
{
	unsigned long long uword = (unsigned long long)word;
	unsigned int ret = 0;
	while (uword >>= 1)
		ret++;
	return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(float word)
{
	unsigned long iword = *((int *)&word);
	unsigned int ret = 0;
	while (iword >>= 1)
		ret++;
	return ret > 64 ? 0 : ret;
}

//MAKE_UNSIGNED

__device__ __host__ __forceinline__ unsigned int MAKE_UNSIGNED(int source)
{
	unsigned int ret = (unsigned int) source;	
	return ret;
}

__device__ __host__ __forceinline__ unsigned long MAKE_UNSIGNED(long source)
{
	unsigned long ret = (unsigned long)source;
	return ret;
}

__device__ __host__ __forceinline__ unsigned long long MAKE_UNSIGNED(long long source)
{
	unsigned long long ret = (unsigned long long)source;
	return ret;
}

__device__ __host__ __forceinline__ unsigned int MAKE_UNSIGNED(unsigned int source)
{
	unsigned int ret = (unsigned int)source;
	return ret;
}

__device__ __host__ __forceinline__ unsigned long MAKE_UNSIGNED(unsigned long source)
{
	unsigned long ret = (unsigned long)source;
	return ret;
}

__device__ __host__ __forceinline__ unsigned long long MAKE_UNSIGNED(unsigned long long source)
{
	unsigned long long ret = (unsigned long long)source;
	return ret;
}

__device__ __host__ __forceinline__ unsigned long MAKE_UNSIGNED(float source)
{
	unsigned long ret = *((int *)&source);
	return ret;
}

__host__ __device__
inline int ALT_BITLEN( int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
}
