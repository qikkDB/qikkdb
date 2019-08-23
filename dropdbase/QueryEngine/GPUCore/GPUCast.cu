#include "GPUCast.cuh"

__device__ int32_t CastIntegral(char* str, int32_t length, int32_t base)
{
    int32_t out = 0;
    int32_t order = 1;
    int32_t sign = 1;
    int32_t numBoundary = 0;

    if (str[0] == '-')
    {
        sign = -1;
        numBoundary = 1;
    }

    for (int32_t i = length - 1; i >= numBoundary; i--)
    {
        out += (str[i] - '0') * order;
        order *= base;
    }
    return sign * out;
}

template <>
__device__ int32_t CastOperations::FromString::operator()<int32_t>(char* str, int32_t length) const
{
    return CastIntegral(str, length);
}

__device__ double CastDecimal(char* str, int32_t length)
{
    double out = 0;
    int decimalPart = 0;

    int c;
    while (((c = *str++) >= '0' && c <= '9') && length > 0)
    {
        out *= 10 + (c - '0');
        length--;
    }

    if ((c == '.' || c == ',') && length > 0)
    {
        length--;
        while (((c = *str++) >= '0' && c <= '9') && length > 0)
        {
            out *= 10 + (c - '0');
            decimal_part--;
            length--;
        }
    }

    else if ((c == 'e' || c == 'E') && length > 0)
    {
        length--;
        int sign = 1;
        int aferEPart = 0;

		c = *s++;
        if (c == '-')
        {
            sign = -1;
            length--;
        }

		else if (c == '+')
        {
            length--;
        }

        while (((c == *str++) >= '0' && c <= '9') && length > 0)
        {
            afterEPart = afterEPart * 10 + (c - '0');
            length--;
        }

        decimalPart += afterEPart * sign;
    }

    while (decimalPart > 0)
    {
        out *= 10;
        decimalPart--;
	}

	while (decimalPart < 0)
    {
        out *= 0.1;
        decimalPart++;
    }

    return out;
}