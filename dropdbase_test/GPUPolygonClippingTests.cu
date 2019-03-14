#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUPolygonClipping.cuh"

#include "gtest/gtest.h"

TEST(GPUPolygonClippingTests, PoygonTest)
{
    Context::getInstance();

    // Input polygons
    NativeGeoPoint poly1[] = {{181, 270}, {85, 418},  {171, 477},
                              {491, 365}, {218, 381}, {458, 260}};
    int32_t complexPolygonIdx1[] = {0};
    int32_t complexPolygonCnt1[] = {1};
    int32_t polygonIdx1[] = {0};
    int32_t polygonCnt1[] = {6};

    NativeGeoPoint poly2[] = {{474, 488}, {659, 363}, {255, 283},
                              {56, 340},  {284, 488}, {371, 342}};
    int32_t complexPolygonIdx2[] = {0};
    int32_t complexPolygonCnt2[] = {1};
    int32_t polygonIdx2[] = {0};
    int32_t polygonCnt2[] = {6};

    int32_t dataElementCount = 1;

    // Buffers on the GPU
    GPUMemory::GPUPolygon polygon1;
    GPUMemory::GPUPolygon polygon2;
    GPUMemory::GPUPolygon polygonOut;

    // Malloc the buffers
    // Polygon 1
    GPUMemory::alloc(&polygon1.polyPoints, sizeof(poly1) / sizeof(NativeGeoPoint));
    GPUMemory::alloc(&polygon1.polyIdx, sizeof(complexPolygonIdx1) / sizeof(int32_t));
    GPUMemory::alloc(&polygon1.polyCount, sizeof(complexPolygonCnt1) / sizeof(int32_t));
    GPUMemory::alloc(&polygon1.pointIdx, sizeof(polygonIdx1) / sizeof(int32_t));
    GPUMemory::alloc(&polygon1.pointCount, sizeof(polygonCnt1) / sizeof(int32_t));

    // Polygon 2
    GPUMemory::alloc(&polygon2.polyPoints, sizeof(poly2) / sizeof(NativeGeoPoint));
    GPUMemory::alloc(&polygon2.polyIdx, sizeof(complexPolygonIdx2) / sizeof(int32_t));
    GPUMemory::alloc(&polygon2.polyCount, sizeof(complexPolygonCnt2) / sizeof(int32_t));
    GPUMemory::alloc(&polygon2.pointIdx, sizeof(polygonIdx2) / sizeof(int32_t));
    GPUMemory::alloc(&polygon2.pointCount, sizeof(polygonCnt2) / sizeof(int32_t));

    // Copy data to GPU
    // Polygon 1
    GPUMemory::copyHostToDevice(polygon1.polyPoints, poly1, sizeof(poly1) / sizeof(NativeGeoPoint));
    GPUMemory::copyHostToDevice(polygon1.polyIdx, complexPolygonIdx1,
                                sizeof(complexPolygonIdx1) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon1.polyCount, complexPolygonCnt1,
                                sizeof(complexPolygonCnt1) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon1.pointIdx, polygonIdx1, sizeof(polygonIdx1) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon1.pointCount, polygonCnt1, sizeof(polygonCnt1) / sizeof(int32_t));

    // Polygon 2
    GPUMemory::copyHostToDevice(polygon2.polyPoints, poly2, sizeof(poly2) / sizeof(NativeGeoPoint));
    GPUMemory::copyHostToDevice(polygon2.polyIdx, complexPolygonIdx2,
                                sizeof(complexPolygonIdx2) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon2.polyCount, complexPolygonCnt2,
                                sizeof(complexPolygonCnt2) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon2.pointIdx, polygonIdx2, sizeof(polygonIdx2) / sizeof(int32_t));
    GPUMemory::copyHostToDevice(polygon2.pointCount, polygonCnt2, sizeof(polygonCnt2) / sizeof(int32_t));

    // Launch intersect
    GPUPolygonClip::ColCol<PolygonFunctions::polyIntersect>(polygonOut, polygon1, polygon2, dataElementCount);

    // TODO Copy back results

    // TODO Print results

    // Free buffers
    // Polygon 1
    GPUMemory::free(polygon1.polyPoints);
    GPUMemory::free(polygon1.polyIdx);
    GPUMemory::free(polygon1.polyCount);
    GPUMemory::free(polygon1.pointIdx);
    GPUMemory::free(polygon1.pointCount);

    // Polygon 2
    GPUMemory::free(polygon2.polyPoints);
    GPUMemory::free(polygon2.polyIdx);
    GPUMemory::free(polygon2.polyCount);
    GPUMemory::free(polygon2.pointIdx);
    GPUMemory::free(polygon2.pointCount);
}