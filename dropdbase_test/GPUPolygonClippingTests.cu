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
                              {491, 365}, {218, 381}, {458, 260}, 
		{270, 370}, {450, 334}, {150, 300}, {200, 450}
    };
    int32_t complexPolygonIdx1[] = {0,1};
    int32_t complexPolygonCnt1[] = {1,1};
    int32_t polygonIdx1[] = {0,6};
    int32_t polygonCnt1[] = {6,4};

    NativeGeoPoint poly2[] = {{474, 488}, {659, 363}, {255, 283},
                              {56, 340},  {284, 488}, {371, 342},
	{450, 420},{300, 300}, {344, 450}};
    int32_t complexPolygonIdx2[] = {0,1};
    int32_t complexPolygonCnt2[] = {1,1};
    int32_t polygonIdx2[] = {0,6};
    int32_t polygonCnt2[] = {6,3};

	NativeGeoPoint polyOutData[] = {
        {153.76, 312.00}, {112.01, 376.36}, {233.41, 455.16}, {322.06, 424.13}, {352.46, 373.12},
        {218.00, 381.00}, {368.00, 305.38}, {255.00, 283.00}, {317.73, 360.45}, {364.00, 351.20}, {324.76, 319.81}};
	int32_t complexPolygonIdxOut[] = {0, 2};
    int32_t complexPolygonCntOut[] = {2, 1};
    int32_t polygonIdxOut[] = {0, 8, 11};
    int32_t polygonCntOut[] = {8, 3, 4};

    int32_t dataElementCount = 2;

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
    GPUPolygonClipping::ColCol<PolygonFunctions::polyIntersect>(polygonOut, polygon1, polygon2, dataElementCount);

	/*
    // Copy back results and compare them
    NativeGeoPoint* res = new NativeGeoPoint[pointOutCount];
    int32_t* complexPolygonIdxRes = new int32_t[dataElementCount];
    int32_t* complexPolygonCntRes = new int32_t[dataElementCount];
    int32_t* polygonIdxRes = new int32_t[complexPolygonOutCount];
    int32_t* polygonCntRes = new int32_t[complexPolygonOutCount];

    GPUMemory::copyDeviceToHost(res, polygonOut.polyPoints, pointOutCount);
    GPUMemory::copyDeviceToHost(complexPolygonIdxRes, polygonOut.polyIdx, dataElementCount);
    GPUMemory::copyDeviceToHost(complexPolygonCntRes, polygonOut.polyCount, dataElementCount);
    GPUMemory::copyDeviceToHost(polygonIdxRes, polygonOut.pointIdx, complexPolygonOutCount);
    GPUMemory::copyDeviceToHost(polygonCntRes, polygonOut.pointCount, complexPolygonOutCount);

    for (int s = 0; s < pointOutCount; s++)
    {
        printf("[%.2f,%.2f],\n", res[s].latitude, res[s].longitude);
    }

    for (int s = 0; s < dataElementCount; s++)
    {
        printf("%d\n", complexPolygonCntRes[s]);
    }

    for (int s = 0; s < dataElementCount; s++)
    {
        printf("%d\n", complexPolygonIdxRes[s]);
    }

    for (int s = 0; s < complexPolygonOutCount; s++)
    {
        printf("%d,\n", polygonCntRes[s]);
    }

    for (int s = 0; s < complexPolygonOutCount; s++)
    {
        printf("%d,\n", polygonIdxRes[s]);
    }

    delete[] res;
    delete[] complexPolygonIdxRes;
    delete[] complexPolygonCntRes;
    delete[] polygonIdxRes;
    delete[] polygonCntRes;
	*/

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

	// Polygon out
    GPUMemory::free(polygonOut.polyPoints);
    GPUMemory::free(polygonOut.polyIdx);
    GPUMemory::free(polygonOut.polyCount);
    GPUMemory::free(polygonOut.pointIdx);
    GPUMemory::free(polygonOut.pointCount);

    // Fail assert
    ASSERT_EQ(0, 1);
}