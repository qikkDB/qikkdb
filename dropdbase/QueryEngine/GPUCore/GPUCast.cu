#include "GPUCast.cuh"

// Cast single geo point matching 'spaces decimal spaces decimal spaces' pattern e.g. ' 12.3  23.4 '
__device__ NativeGeoPoint CastNativeGeoPoint(char* str, int32_t length)
{
    int32_t latStartIdx = 0;

    while ((*(str + latStartIdx) == ' ' || *(str + latStartIdx) == '\t') && latStartIdx < length)
    {
        latStartIdx++;
    }

    int32_t spaceStartIdx = latStartIdx;

    while ((*(str + spaceStartIdx) != ' ' && *(str + spaceStartIdx) != '\t') && spaceStartIdx < length)
    {
        spaceStartIdx++;
    }

    int32_t lonStartIdx = spaceStartIdx;

    while ((*(str + lonStartIdx) == ' ' || *(str + lonStartIdx) == '\t') && lonStartIdx < length)
    {
        lonStartIdx++;
    }

    int32_t lonEndIdx = lonStartIdx;

    while ((*(str + lonEndIdx) != ' ' && *(str + lonEndIdx) != '\t') && lonEndIdx < length)
    {
        lonEndIdx++;
    }

    NativeGeoPoint geoPoint;
    geoPoint.latitude = CastDecimal<float>(str + latStartIdx, spaceStartIdx - latStartIdx);
    geoPoint.longitude = CastDecimal<float>(str + lonStartIdx, lonEndIdx - lonStartIdx);

    return geoPoint;
}

// Cast single WKT point matching e.g. 'POINT(122.123 123.23)'
__device__ NativeGeoPoint CastWKTPoint(char* str, int32_t length)
{
    // + 6 represents length of "POINT(" string and lenght - 7 accounts also for the trailing parenthesis ")"
    return CastNativeGeoPoint(str + 6, length - 7);
}

template <>
__device__ int32_t CastOperations::FromString::operator()<int32_t>(char* str, int32_t length) const
{
    return CastDecimal<int32_t>(str, length);
}

template <>
__device__ int64_t CastOperations::FromString::operator()<int64_t>(char* str, int32_t length) const
{
    return CastDecimal<int64_t>(str, length);
}

template <>
__device__ float CastOperations::FromString::operator()<float>(char* str, int32_t length) const
{
    return CastDecimal<float>(str, length);
}

template <>
__device__ double CastOperations::FromString::operator()<double>(char* str, int32_t length) const
{
    return CastDecimal<double>(str, length);
}

template <>
__device__ NativeGeoPoint CastOperations::FromString::operator()<NativeGeoPoint>(char* str, int32_t length) const
{
    return CastWKTPoint(str, length);
}