#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../qikkDB/QueryEngine/Context.h"
#include "../qikkDB/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../qikkDB/QueryEngine/GPUCore/GPUPolygonClipping.cuh"

#include "gtest/gtest.h"

int32_t dataElementCount = 3;

std::vector<int32_t> polyApolyIdxConst = {3};
std::vector<int32_t> polyApointsIdxConst = {4, 8, 12};
std::vector<NativeGeoPoint> polyApolyPointsConst = {{4.50, 5.50},  {6.00, 5.50},
                                                    {6.00, 4.50},  {4.50, 4.50},

                                                    {10.00, 0.00}, {0.00, 0.00},
                                                    {0.00, 10.00}, {10.00, 10.00},

                                                    {7.00, 7.00},  {3.00, 7.00},
                                                    {3.00, 3.00},  {7.00, 3.00}};

std::vector<int32_t> polyBpolyIdxConst = {2};
std::vector<int32_t> polyBpointsIdxConst = {6, 12};
std::vector<NativeGeoPoint> polyBpolyPointsConst = {{-5.12, 4.59}, {0.42, -5.63}, {3.86, 0.41},
                                                    {2.06, 3.75},  {1.22, 6.83},  {-4.60, 6.45},

                                                    {-3.32, 4.11}, {0.92, 3.69},  {2.26, 0.21},
                                                    {0.00, -3.00}, {0.48, 1.65},  {-2.94, 2.27}};

std::vector<int32_t> polyApolyIdx = {3, 4, 7};
std::vector<int32_t> polyApointsIdx = {4, 8, 12, 15, 22, 27, 31};
std::vector<NativeGeoPoint> polyApolyPoints = {
    {4.50, 5.50},   {6.00, 5.50},   {6.00, 4.50},  {4.50, 4.50},

    {10.00, 0.00},  {0.00, 0.00},   {0.00, 10.00}, {10.00, 10.00},

    {7.00, 7.00},   {3.00, 7.00},   {3.00, 3.00},  {7.00, 3.00},

    {0.00, 0.00},   {1.00, 0.00},   {0.50, 1.00},

    {-6.31, -1.49}, {-4.00, 5.00},  {2.13, 6.03},  {4.90, 2.23},   {-0.52, -0.49},
    {3.88, -3.45},  {-4.33, -3.89},

    {-3.77, 2.88},  {1.12, 5.24},   {3.52, 2.73},  {-0.92, 0.45},  {-2.82, -2.57},

    {-2.52, 1.91},  {0.96, 4.25},   {2.16, 2.81},  {-1.98, 0.43}};

std::vector<int32_t> polyBpolyIdx = {2, 3, 5};
std::vector<int32_t> polyBpointsIdx = {3, 7, 11, 17, 23};
std::vector<NativeGeoPoint> polyBpolyPoints = {
    {13.00, 5.50}, {13.00, 4.50}, {5.00, 5.00},

    {4.00, 4.00},  {15.00, 4.00}, {15.00, 6.00}, {4.00, 6.00},

    {-0.50, 0.40}, {1.00, 0.40},  {1.00, 0.60},  {-0.50, 0.60},

    {-5.12, 4.59}, {0.42, -5.63}, {3.86, 0.41},  {2.06, 3.75},  {1.22, 6.83}, {-4.60, 6.45},

    {-3.32, 4.11}, {0.92, 3.69},  {2.26, 0.21},  {0.00, -3.00}, {0.48, 1.65}, {-2.94, 2.27}};


template <typename OP>
void polyTest(std::vector<int32_t>& polyApolyIdx,
              std::vector<int32_t>& polyApointsIdx,
              std::vector<NativeGeoPoint>& polyApolyPoints,
              std::vector<int32_t>& polyBpolyIdx,
              std::vector<int32_t>& polyBpointsIdx,
              std::vector<NativeGeoPoint>& polyBpolyPoints,
              std::vector<int32_t>& outPolyIdx,
              std::vector<int32_t>& outPointIdx,
              std::vector<NativeGeoPoint>& outPolyPoints,
              bool isAConst,
              bool isBConst,
              int32_t dataElementCount)
{
    // Alloc the GPU structures
    GPUMemory::GPUPolygon polygonA;
    GPUMemory::GPUPolygon polygonB;

    GPUMemory::alloc(&polygonA.polyIdx, polyApolyIdx.size());
    GPUMemory::alloc(&polygonA.pointIdx, polyApointsIdx.size());
    GPUMemory::alloc(&polygonA.polyPoints, polyApolyPoints.size());

    GPUMemory::alloc(&polygonB.polyIdx, polyBpolyIdx.size());
    GPUMemory::alloc(&polygonB.pointIdx, polyBpointsIdx.size());
    GPUMemory::alloc(&polygonB.polyPoints, polyBpolyPoints.size());

    // Copy the input data to the GPU
    GPUMemory::copyHostToDevice(polygonA.polyIdx, &polyApolyIdx[0], polyApolyIdx.size());
    GPUMemory::copyHostToDevice(polygonA.pointIdx, &polyApointsIdx[0], polyApointsIdx.size());
    GPUMemory::copyHostToDevice(polygonA.polyPoints, &polyApolyPoints[0], polyApolyPoints.size());

    GPUMemory::copyHostToDevice(polygonB.polyIdx, &polyBpolyIdx[0], polyBpolyIdx.size());
    GPUMemory::copyHostToDevice(polygonB.pointIdx, &polyBpointsIdx[0], polyBpointsIdx.size());
    GPUMemory::copyHostToDevice(polygonB.polyPoints, &polyBpolyPoints[0], polyBpolyPoints.size());

    // Perform the polygon intersect
    GPUMemory::GPUPolygon polygonOut; // This needs to be empty

    if (!isAConst && !isBConst)
    {
        GPUPolygonClipping::clip<OP>(polygonOut, polygonA, polygonB, dataElementCount, dataElementCount);
    }
    else if (!isAConst && isBConst)
    {
        GPUPolygonClipping::clip<OP>(polygonOut, polygonA, polygonB, dataElementCount, 1);
    }
    else if (isAConst && !isBConst)
    {
        GPUPolygonClipping::clip<OP>(polygonOut, polygonA, polygonB, 1, dataElementCount);
    }
    else if (isAConst && isBConst)
    {
        GPUPolygonClipping::clip<OP>(polygonOut, polygonA, polygonB, 1, 1);
        ;
    }

    // Copy back the results
    outPolyIdx.resize(dataElementCount);
    GPUMemory::copyDeviceToHost(&outPolyIdx[0], polygonOut.polyIdx, outPolyIdx.size());

    // Check if a result set is empty
    if (outPolyIdx[dataElementCount - 1] <= 0)
    {
        outPointIdx.resize(0);
        outPolyPoints.resize(0);
    }
    else
    {
        outPointIdx.resize(outPolyIdx[dataElementCount - 1]);
        GPUMemory::copyDeviceToHost(&outPointIdx[0], polygonOut.pointIdx, outPointIdx.size());

        outPolyPoints.resize(outPointIdx[outPolyIdx[dataElementCount - 1] - 1]);
        GPUMemory::copyDeviceToHost(&outPolyPoints[0], polygonOut.polyPoints, outPolyPoints.size());
    }

    // Free the polygons
    GPUMemory::free(polygonA);
    GPUMemory::free(polygonB);
    GPUMemory::free(polygonOut);
}

float roundCustom(float var)
{
    float value = (int)(var * 100 + .5);
    return (float)value / 100;
}

void polyCompare(std::vector<int32_t>& outPolyIdx,
                 std::vector<int32_t>& outPointIdx,
                 std::vector<NativeGeoPoint>& outPolyPoints,
                 std::vector<int32_t>& outPolyIdxCorrect,
                 std::vector<int32_t>& outPointIdxCorrect,
                 std::vector<NativeGeoPoint>& outPolyPointsCorrect)
{
    ASSERT_EQ(outPolyIdx.size(), outPolyIdxCorrect.size());
    for (int32_t i = 0; i < outPolyIdx.size(); i++)
    {
        ASSERT_EQ(outPolyIdx[i], outPolyIdxCorrect[i]);
    }

    ASSERT_EQ(outPointIdx.size(), outPointIdxCorrect.size());
    for (int32_t i = 0; i < outPointIdx.size(); i++)
    {
        ASSERT_EQ(outPointIdx[0], outPointIdxCorrect[0]);
    }

    ASSERT_EQ(outPolyPoints.size(), outPolyPointsCorrect.size());
    for (int32_t i = 0; i < outPolyPoints.size(); i++)
    {
        ASSERT_FLOAT_EQ(roundCustom(outPolyPoints[i].latitude), roundCustom(outPolyPointsCorrect[i].latitude));
        ASSERT_FLOAT_EQ(roundCustom(outPolyPoints[i].longitude), roundCustom(outPolyPointsCorrect[i].longitude));
    }
}


void printPolygonAsList(std::vector<int32_t> polyIdx,
                        std::vector<int32_t> pointIdx,
                        std::vector<NativeGeoPoint> polyPoints,
                        bool showIndex = false)
{
    for (int32_t i = 0; i < polyIdx.size(); i++)
    {
        if (showIndex)
            printf("%2d : %2d\n", i, polyIdx[i]);
        else
            printf("%2d, ", polyIdx[i]);
    }
    printf("\n");

    for (int32_t i = 0; i < pointIdx.size(); i++)
    {
        if (showIndex)
            printf("%2d : %2d\n", i, pointIdx[i]);
        else
            printf("%2d, ", pointIdx[i]);
    }
    printf("\n");

    for (int32_t i = 0; i < polyPoints.size(); i++)
    {
        if (showIndex)
            printf("%2d : %.2f, %.2f\n", i, polyPoints[i].latitude, polyPoints[i].longitude);
        else
            printf("{%.2f, %.2f},\n", polyPoints[i].latitude, polyPoints[i].longitude);
    }
    printf("\n");
}

// Polygon visualizer: https://www.geogebra.org/classic
void printPolygonAsGeoGebraPolygons(std::vector<int32_t> polyIdx,
                                    std::vector<int32_t> pointIdx,
                                    std::vector<NativeGeoPoint> polyPoints,
                                    char startLabel = 'A')
{
    char labelOuter = startLabel;
    for (int32_t i = 0; i < polyIdx.size(); i++)
    {
        char labelInner = startLabel;

        int32_t i_polyIdx = (i == 0) ? 0 : polyIdx[i - 1];
        int32_t i_polyCount = polyIdx[i] - ((i == 0) ? 0 : polyIdx[i - 1]);

        for (int32_t p = i_polyIdx; p < (i_polyIdx + i_polyCount); p++)
        {
            printf("%c%c=Polygon(\n", labelOuter, labelInner);

            int32_t p_pointIdx = (p == 0) ? 0 : pointIdx[p - 1];
            int32_t p_pointCount = pointIdx[p] - ((p == 0) ? 0 : pointIdx[p - 1]);

            for (int32_t point = p_pointIdx; point < (p_pointIdx + p_pointCount); point++)
            {
                printf("(%.2f, %.2f)", polyPoints[point].latitude, polyPoints[point].longitude);

                if (point == (p_pointIdx + p_pointCount - 1))
                {
                    printf(")\n");
                    break;
                }

                printf(",\n");
            }

            labelInner++;
            printf("\n");
        }

        labelOuter++;
        printf("\n");
    }
}

TEST(GPUPolygonClippingTests, IntersectColColTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdx, polyApointsIdx, polyApolyPoints, polyBpolyIdx,
                                              polyBpointsIdx, polyBpolyPoints, outPolyIdx, outPointIdx,
                                              outPolyPoints, false, false, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {3, 4, 9};
    std::vector<int32_t> outPointIdxCorrect = {7, 11, 15, 19, 29, 33, 44, 49, 53};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {6.00, 4.94},   {6.00, 4.50},   {4.50, 4.50},  {4.50, 5.50},  {6.00, 5.50},  {6.00, 5.06},
        {5.00, 5.00},   {10.00, 4.69},  {10.00, 4.00}, {7.00, 4.00},  {7.00, 4.88},  {10.00, 5.31},
        {10.00, 6.00},  {7.00, 6.00},   {7.00, 5.13},  {0.80, 0.40},  {0.70, 0.60},  {0.30, 0.60},
        {0.20, 0.40},   {-4.53, 3.51},  {-4.00, 5.00}, {1.47, 5.92},  {1.87, 4.46},  {1.12, 5.24},
        {-1.58, 3.94},  {-3.32, 4.11},  {-3.13, 3.19}, {-3.77, 2.88}, {-3.57, 1.73}, {3.31, 1.43},
        {2.04, 0.79},   {1.66, 1.77},   {2.81, 2.36},  {2.28, -2.37}, {1.03, -1.53}, {0.00, -3.00},
        {0.21, -0.98},  {-0.52, -0.49}, {0.30, -0.08}, {0.43, 1.14},  {-0.92, 0.45}, {-1.97, -1.22},
        {-0.63, -3.69}, {1.59, -3.57},  {0.23, 3.76},  {0.96, 4.25},  {2.16, 2.81},  {1.42, 2.39},
        {0.92, 3.69},   {-2.19, 2.13},  {-2.52, 1.91}, {-1.98, 0.43}, {0.22, 1.70}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, UnionColColTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the union test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdx, polyApointsIdx, polyApolyPoints,
                                          polyBpolyIdx, polyBpointsIdx, polyBpolyPoints, outPolyIdx,
                                          outPointIdx, outPolyPoints, false, false, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {4, 5, 11};
    std::vector<int32_t> outPointIdxCorrect = {4, 8, 16, 24, 35, 51, 56, 60, 63, 68, 73};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {6.00, 4.94},   {6.00, 5.06},   {7.00, 5.13},   {7.00, 4.88},  {10.00, 4.69}, {10.00, 5.31},
        {13.00, 5.50},  {13.00, 4.50},  {10.00, 4.00},  {10.00, 0.00}, {0.00, 0.00},  {0.00, 10.00},
        {10.00, 10.00}, {10.00, 6.00},  {15.00, 6.00},  {15.00, 4.00}, {7.00, 4.00},  {7.00, 3.00},
        {3.00, 3.00},   {3.00, 7.00},   {7.00, 7.00},   {7.00, 6.00},  {4.00, 6.00},  {4.00, 4.00},
        {0.80, 0.40},   {1.00, 0.00},   {0.00, 0.00},   {0.20, 0.40},  {-0.50, 0.40}, {-0.50, 0.60},
        {0.30, 0.60},   {0.50, 1.00},   {0.70, 0.60},   {1.00, 0.60},  {1.00, 0.40},  {-4.53, 3.51},
        {-6.31, -1.49}, {-4.33, -3.89}, {-0.63, -3.69}, {0.42, -5.63}, {1.59, -3.57}, {3.88, -3.45},
        {2.28, -2.37},  {3.86, 0.41},   {3.31, 1.43},   {4.90, 2.23},  {2.13, 6.03},  {1.47, 5.92},
        {1.22, 6.83},   {-4.60, 6.45},  {-5.12, 4.59},  {2.04, 0.79},  {0.30, -0.08}, {0.21, -0.98},
        {1.03, -1.53},  {2.26, 0.21},   {1.87, 4.46},   {3.52, 2.73},  {2.81, 2.36},  {2.06, 3.75},
        {-1.97, -1.22}, {-2.82, -2.57}, {-3.57, 1.73},  {-1.58, 3.94}, {-3.13, 3.19}, {-2.94, 2.27},
        {-2.19, 2.13},  {0.23, 3.76},   {1.66, 1.77},   {0.43, 1.14},  {0.48, 1.65},  {0.22, 1.70},
        {1.42, 2.39}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, IntersectColConstTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdx, polyApointsIdx, polyApolyPoints, polyBpolyIdxConst,
                                              polyBpointsIdxConst, polyBpolyPointsConst, outPolyIdx,
                                              outPointIdx, outPolyPoints, false, true, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {2, 3, 8};
    std::vector<int32_t> outPointIdxCorrect = {9, 13, 16, 26, 30, 41, 46, 50};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {3.63, 0.00},  {2.11, 0.00},  {2.26, 0.21},   {0.92, 3.69},   {0.00, 3.78},   {0.00, 6.75},
        {1.22, 6.83},  {2.06, 3.75},  {3.86, 0.41},   {0.31, 0.00},   {0.00, 0.00},   {0.00, 1.74},
        {0.48, 1.65},  {0.31, 0.00},  {0.00, 0.00},   {0.39, 0.78},   {-4.53, 3.51},  {-4.00, 5.00},
        {1.47, 5.92},  {1.87, 4.46},  {1.12, 5.24},   {-1.58, 3.94},  {-3.32, 4.11},  {-3.13, 3.19},
        {-3.77, 2.88}, {-3.57, 1.73}, {3.31, 1.43},   {2.04, 0.79},   {1.66, 1.77},   {2.81, 2.36},
        {2.28, -2.37}, {1.03, -1.53}, {0.00, -3.00},  {0.21, -0.98},  {-0.52, -0.49}, {0.30, -0.08},
        {0.43, 1.14},  {-0.92, 0.45}, {-1.97, -1.22}, {-0.63, -3.69}, {1.59, -3.57},  {0.23, 3.76},
        {0.96, 4.25},  {2.16, 2.81},  {1.42, 2.39},   {0.92, 3.69},   {-2.19, 2.13},  {-2.52, 1.91},
        {-1.98, 0.43}, {0.22, 1.70}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, UnionColConstTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdx, polyApointsIdx, polyApolyPoints, polyBpolyIdxConst,
                                          polyBpointsIdxConst, polyBpolyPointsConst, outPolyIdx,
                                          outPointIdx, outPolyPoints, false, true, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {5, 7, 13};
    std::vector<int32_t> outPointIdxCorrect = {4, 8, 16, 19, 23, 29, 39, 55, 60, 64, 67, 72, 77};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {4.50, 5.50},   {6.00, 5.50},  {6.00, 4.50},  {4.50, 4.50},  {7.00, 7.00},   {3.00, 7.00},
        {3.00, 3.00},   {7.00, 3.00},  {3.63, 0.00},  {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00},
        {0.00, 6.75},   {-4.60, 6.45}, {-5.12, 4.59}, {0.42, -5.63}, {2.11, 0.00},   {0.31, 0.00},
        {0.00, -3.00},  {0.00, 3.78},  {0.00, 1.74},  {-2.94, 2.27}, {-3.32, 4.11},  {-5.12, 4.59},
        {0.42, -5.63},  {3.86, 0.41},  {2.06, 3.75},  {1.22, 6.83},  {-4.60, 6.45},  {0.31, 0.00},
        {1.00, 0.00},   {0.50, 1.00},  {0.39, 0.78},  {0.48, 1.65},  {-2.94, 2.27},  {-3.32, 4.11},
        {0.92, 3.69},   {2.26, 0.21},  {0.00, -3.00}, {-4.53, 3.51}, {-6.31, -1.49}, {-4.33, -3.89},
        {-0.63, -3.69}, {0.42, -5.63}, {1.59, -3.57}, {3.88, -3.45}, {2.28, -2.37},  {3.86, 0.41},
        {3.31, 1.43},   {4.90, 2.23},  {2.13, 6.03},  {1.47, 5.92},  {1.22, 6.83},   {-4.60, 6.45},
        {-5.12, 4.59},  {2.04, 0.79},  {0.30, -0.08}, {0.21, -0.98}, {1.03, -1.53},  {2.26, 0.21},
        {1.87, 4.46},   {3.52, 2.73},  {2.81, 2.36},  {2.06, 3.75},  {-1.97, -1.22}, {-2.82, -2.57},
        {-3.57, 1.73},  {-1.58, 3.94}, {-3.13, 3.19}, {-2.94, 2.27}, {-2.19, 2.13},  {0.23, 3.76},
        {1.66, 1.77},   {0.43, 1.14},  {0.48, 1.65},  {0.22, 1.70},  {1.42, 2.39}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, IntersectConstColTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                              polyBpolyIdx, polyBpointsIdx, polyBpolyPoints, outPolyIdx,
                                              outPointIdx, outPolyPoints, true, false, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {3, 4, 6};
    std::vector<int32_t> outPointIdxCorrect = {7, 11, 15, 19, 28, 32};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {6.00, 4.94},  {6.00, 4.50},  {4.50, 4.50},  {4.50, 5.50}, {6.00, 5.50}, {6.00, 5.06},
        {5.00, 5.00},  {10.00, 4.69}, {10.00, 4.00}, {7.00, 4.00}, {7.00, 4.88}, {10.00, 5.31},
        {10.00, 6.00}, {7.00, 6.00},  {7.00, 5.13},  {0.00, 0.40}, {0.00, 0.60}, {1.00, 0.60},
        {1.00, 0.40},  {3.63, 0.00},  {2.11, 0.00},  {2.26, 0.21}, {0.92, 3.69}, {0.00, 3.78},
        {0.00, 6.75},  {1.22, 6.83},  {2.06, 3.75},  {3.86, 0.41}, {0.31, 0.00}, {0.00, 0.00},
        {0.00, 1.74},  {0.48, 1.65}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, UnionConstColTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                          polyBpolyIdx, polyBpointsIdx, polyBpolyPoints, outPolyIdx,
                                          outPointIdx, outPolyPoints, true, false, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {4, 7, 12};
    std::vector<int32_t> outPointIdxCorrect = {4, 8, 16, 24, 28, 32, 40, 44, 48, 56, 59, 63};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {6.00, 4.94},   {6.00, 5.06},  {7.00, 5.13},   {7.00, 4.88},  {10.00, 4.69}, {10.00, 5.31},
        {13.00, 5.50},  {13.00, 4.50}, {10.00, 4.00},  {10.00, 0.00}, {0.00, 0.00},  {0.00, 10.00},
        {10.00, 10.00}, {10.00, 6.00}, {15.00, 6.00},  {15.00, 4.00}, {7.00, 4.00},  {7.00, 3.00},
        {3.00, 3.00},   {3.00, 7.00},  {7.00, 7.00},   {7.00, 6.00},  {4.00, 6.00},  {4.00, 4.00},
        {4.50, 5.50},   {6.00, 5.50},  {6.00, 4.50},   {4.50, 4.50},  {7.00, 7.00},  {3.00, 7.00},
        {3.00, 3.00},   {7.00, 3.00},  {0.00, 0.40},   {0.00, 0.00},  {10.00, 0.00}, {10.00, 10.00},
        {0.00, 10.00},  {0.00, 0.60},  {-0.50, 0.60},  {-0.50, 0.40}, {4.50, 5.50},  {6.00, 5.50},
        {6.00, 4.50},   {4.50, 4.50},  {7.00, 7.00},   {3.00, 7.00},  {3.00, 3.00},  {7.00, 3.00},
        {3.63, 0.00},   {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00}, {0.00, 6.75},  {-4.60, 6.45},
        {-5.12, 4.59},  {0.42, -5.63}, {2.11, 0.00},   {0.31, 0.00},  {0.00, -3.00}, {0.00, 3.78},
        {0.00, 1.74},   {-2.94, 2.27}, {-3.32, 4.11}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, IntersectConstConstTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                              polyBpolyIdxConst, polyBpointsIdxConst, polyBpolyPointsConst,
                                              outPolyIdx, outPointIdx, outPolyPoints, true, true, 1);

    std::vector<int32_t> outPolyIdxCorrect = {2};
    std::vector<int32_t> outPointIdxCorrect = {9, 13};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {{3.63, 0.00}, {2.11, 0.00}, {2.26, 0.21},
                                                        {0.92, 3.69}, {0.00, 3.78}, {0.00, 6.75},
                                                        {1.22, 6.83}, {2.06, 3.75}, {3.86, 0.41},
                                                        {0.31, 0.00}, {0.00, 0.00}, {0.00, 1.74},
                                                        {0.48, 1.65}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, UnionConstConstTest)
{
    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                          polyBpolyIdxConst, polyBpointsIdxConst, polyBpolyPointsConst,
                                          outPolyIdx, outPointIdx, outPolyPoints, true, true, 1);

    std::vector<int32_t> outPolyIdxCorrect = {5};
    std::vector<int32_t> outPointIdxCorrect = {4, 8, 16, 19, 23};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {4.50, 5.50},  {6.00, 5.50},  {6.00, 4.50},  {4.50, 4.50},  {7.00, 7.00},   {3.00, 7.00},
        {3.00, 3.00},  {7.00, 3.00},  {3.63, 0.00},  {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00},
        {0.00, 6.75},  {-4.60, 6.45}, {-5.12, 4.59}, {0.42, -5.63}, {2.11, 0.00},   {0.31, 0.00},
        {0.00, -3.00}, {0.00, 3.78},  {0.00, 1.74},  {-2.94, 2.27}, {-3.32, 4.11}};

    // Check the results
    polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);
}

TEST(GPUPolygonClippingTests, IntersectColColEmptyResultSetTest)
{
    std::vector<int32_t> polyApolyIdx = {1};
    std::vector<int32_t> polyApointsIdx = {4};
    std::vector<NativeGeoPoint> polyApolyPoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    std::vector<int32_t> polyBpolyIdx = {1};
    std::vector<int32_t> polyBpointsIdx = {4};
    std::vector<NativeGeoPoint> polyBpolyPoints = {{2.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {2.0, 3.0}};

    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdx, polyApointsIdx, polyApolyPoints,
                                              polyBpolyIdx, polyBpointsIdx, polyBpolyPoints,
                                              outPolyIdx, outPointIdx, outPolyPoints, false, false, 1);

    ASSERT_EQ(outPolyIdx.size(), 1);
    ASSERT_EQ(outPolyIdx[0], 0);

    ASSERT_EQ(outPointIdx.size(), 0);

    ASSERT_EQ(outPolyPoints.size(), 0);
}