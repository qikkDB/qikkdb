#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUPolygonClipping.cuh"

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
        GPUPolygonClipping::ColCol<OP>(polygonOut, polygonA, polygonB, dataElementCount);
    }
    else if (!isAConst && isBConst)
    {
        GPUPolygonClipping::ColConst<OP>(polygonOut, polygonA, polygonB, dataElementCount);
    }
    else if (isAConst && !isBConst)
    {
        GPUPolygonClipping::ConstCol<OP>(polygonOut, polygonA, polygonB, dataElementCount);
    }
    else if (isAConst && isBConst)
    {
        GPUPolygonClipping::ConstConst<OP>(polygonOut, polygonA, polygonB, dataElementCount);
    }

    // Copy back the results
    outPolyIdx.resize(dataElementCount);
    GPUMemory::copyDeviceToHost(&outPolyIdx[0], polygonOut.polyIdx, outPolyIdx.size());

    outPointIdx.resize(outPolyIdx[dataElementCount - 1]);
    GPUMemory::copyDeviceToHost(&outPointIdx[0], polygonOut.pointIdx, outPointIdx.size());

    outPolyPoints.resize(outPointIdx[outPolyIdx[dataElementCount - 1] - 1]);
    GPUMemory::copyDeviceToHost(&outPolyPoints[0], polygonOut.polyPoints, outPolyPoints.size());

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
    //ASSERT_EQ(outPolyIdx.size(), outPolyIdxCorrect.size());
    for (int32_t i = 0; i < outPolyIdx.size(); i++)
    {
        //ASSERT_EQ(outPolyIdx[i], outPolyIdxCorrect[i]);
    }

    //ASSERT_EQ(outPointIdx.size(), outPointIdxCorrect.size());
    for (int32_t i = 0; i < outPointIdx.size(); i++)
    {
        //ASSERT_EQ(outPointIdx[0], outPointIdxCorrect[0]);
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
            printf("%2d, ", polyIdx[i]);
        else
            printf("%2d : %2d\n", i, polyIdx[i]);
    }
    printf("\n");

    for (int32_t i = 0; i < pointIdx.size(); i++)
    {
        if (showIndex)
            printf("%2d, ", pointIdx[i]);
        else
            printf("%2d : %2d\n", i, pointIdx[i]);
    }
    printf("\n");

    for (int32_t i = 0; i < polyPoints.size(); i++)
    {
        if (showIndex)
            printf("{%.2f, %.2f},\n", polyPoints[i].latitude, polyPoints[i].longitude);
        else
            printf("%2d : %.2f, %.2f\n", i, polyPoints[i].latitude, polyPoints[i].longitude);
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
        {6.000000, 4.937500},   {6.000000, 4.500000},   {4.500000, 4.500000},
        {4.500000, 5.500000},   {6.000000, 5.500000},   {6.000000, 5.062500},
        {5.000000, 5.000000},   {10.000000, 4.687500},  {10.000000, 4.000000},
        {7.000000, 4.000000},   {7.000000, 4.875000},   {10.000000, 5.312500},
        {10.000000, 6.000000},  {7.000000, 6.000000},   {7.000000, 5.125000},
        {0.800000, 0.400000},   {0.700000, 0.600000},   {0.300000, 0.600000},
        {0.200000, 0.400000},   {-4.532012, 3.505300},  {-4.000000, 5.000000},
        {1.468495, 5.918850},   {1.866582, 4.459199},   {1.120000, 5.240000},
        {-1.578781, 3.937521},  {-3.320000, 4.110000},  {-3.129789, 3.188977},
        {-3.770000, 2.880000},  {-3.569482, 1.729657},  {3.309359, 1.431745},
        {2.035682, 0.792556},   {1.657860, 1.773766},   {2.807015, 2.363873},
        {2.276109, -2.371018},  {1.032058, -1.534112},  {0.000000, -3.000000},
        {0.208507, -0.980087},  {-0.520000, -0.490000}, {0.301662, -0.077653},
        {0.427560, 1.141991},   {-0.920000, 0.450000},  {-1.970601, -1.219902},
        {-0.630680, -3.691742}, {1.591744, -3.572635},  {0.228978, 3.758451},
        {0.960000, 4.250000},   {2.160000, 2.810000},   {1.422176, 2.385841},
        {0.920000, 3.690000},   {-2.187495, 2.133581},  {-2.520000, 1.910000},
        {-1.980000, 0.430000},  {0.223174, 1.696559}};


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
        {6.000000, 4.937500},   {6.000000, 5.062500},   {7.000000, 5.125000},
        {7.000000, 4.875000},   {10.000000, 4.687500},  {10.000000, 5.312500},
        {13.000000, 5.500000},  {13.000000, 4.500000},  {10.000000, 4.000000},
        {10.000000, 0.000000},  {0.000000, 0.000000},   {0.000000, 10.000000},
        {10.000000, 10.000000}, {10.000000, 6.000000},  {15.000000, 6.000000},
        {15.000000, 4.000000},  {7.000000, 4.000000},   {7.000000, 3.000000},
        {3.000000, 3.000000},   {3.000000, 7.000000},   {7.000000, 7.000000},
        {7.000000, 6.000000},   {4.000000, 6.000000},   {4.000000, 4.000000},
        {0.800000, 0.400000},   {1.000000, 0.000000},   {0.000000, 0.000000},
        {0.200000, 0.400000},   {-0.500000, 0.400000},  {-0.500000, 0.600000},
        {0.300000, 0.600000},   {0.500000, 1.000000},   {0.700000, 0.600000},
        {1.000000, 0.600000},   {1.000000, 0.400000},   {-4.532012, 3.505300},
        {-6.310000, -1.490000}, {-4.330000, -3.890000}, {-0.630680, -3.691742},
        {0.420000, -5.630000},  {1.591744, -3.572635},  {3.880000, -3.450000},
        {2.276109, -2.371018},  {3.860000, 0.410000},   {3.309359, 1.431745},
        {4.900000, 2.230000},   {2.130000, 6.030000},   {1.468495, 5.918850},
        {1.220000, 6.830000},   {-4.600000, 6.450000},  {-5.120000, 4.590000},
        {2.035682, 0.792556},   {0.301662, -0.077653},  {0.208507, -0.980087},
        {1.032058, -1.534112},  {2.260000, 0.210000},   {1.866582, 4.459199},
        {3.520000, 2.730000},   {2.807015, 2.363873},   {2.060000, 3.750000},
        {-1.970601, -1.219902}, {-2.820000, -2.570000}, {-3.569482, 1.729657},
        {-1.578781, 3.937521},  {-3.129789, 3.188977},  {-2.940000, 2.270000},
        {-2.187495, 2.133581},  {0.228978, 3.758451},   {1.657860, 1.773766},
        {0.427560, 1.141991},   {0.480000, 1.650000},   {0.223174, 1.696559},
        {1.422176, 2.385841}};

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
    // WARNING
    // TODO WARNING - The correct results are for the algorithm WITHOUT considering the non-intersecting
    // polygons/holes - AFTER ADDING THIS FEATURE THE TEST MIGHT BREAK
    // The expected results of this test need to be MODIFIED after the correction is added

    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdx, polyApointsIdx, polyApolyPoints, polyBpolyIdxConst,
                                          polyBpointsIdxConst, polyBpolyPointsConst, outPolyIdx,
                                          outPointIdx, outPolyPoints, false, true, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {3, 4, 10};
    std::vector<int32_t> outPointIdxCorrect = {8, 11, 15, 25, 41, 46, 50, 53, 58, 63};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {3.63, 0.00},  {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00},  {0.00, 6.75},   {-4.60, 6.45},
        {-5.12, 4.59}, {0.42, -5.63}, {2.11, 0.00},   {0.31, 0.00},   {0.00, -3.00},  {0.00, 3.78},
        {0.00, 1.74},  {-2.94, 2.27}, {-3.32, 4.11},  {0.31, 0.00},   {1.00, 0.00},   {0.50, 1.00},
        {0.39, 0.78},  {0.48, 1.65},  {-2.94, 2.27},  {-3.32, 4.11},  {0.92, 3.69},   {2.26, 0.21},
        {0.00, -3.00}, {-4.53, 3.51}, {-6.31, -1.49}, {-4.33, -3.89}, {-0.63, -3.69}, {0.42, -5.63},
        {1.59, -3.57}, {3.88, -3.45}, {2.28, -2.37},  {3.86, 0.41},   {3.31, 1.43},   {4.90, 2.23},
        {2.13, 6.03},  {1.47, 5.92},  {1.22, 6.83},   {-4.60, 6.45},  {-5.12, 4.59},  {2.04, 0.79},
        {0.30, -0.08}, {0.21, -0.98}, {1.03, -1.53},  {2.26, 0.21},   {1.87, 4.46},   {3.52, 2.73},
        {2.81, 2.36},  {2.06, 3.75},  {-1.97, -1.22}, {-2.82, -2.57}, {-3.57, 1.73},  {-1.58, 3.94},
        {-3.13, 3.19}, {-2.94, 2.27}, {-2.19, 2.13},  {0.23, 3.76},   {1.66, 1.77},   {0.43, 1.14},
        {0.48, 1.65},  {0.22, 1.70},  {1.42, 2.39}};

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
    // WARNING
    // TODO WARNING - The correct results are for the algorithm WITHOUT considering the non-intersecting
    // polygons/holes - AFTER ADDING THIS FEATURE THE TEST MIGHT BREAK
    // The expected results of this test need to be MODIFIED after the correction is added

    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                          polyBpolyIdx, polyBpointsIdx, polyBpolyPoints, outPolyIdx,
                                          outPointIdx, outPolyPoints, true, false, dataElementCount);

    std::vector<int32_t> outPolyIdxCorrect = {4, 5, 8};
    std::vector<int32_t> outPointIdxCorrect = {4, 8, 16, 24, 32, 40, 43, 47};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {6.00, 4.94},   {6.00, 5.06},  {7.00, 5.13},  {7.00, 4.88},   {10.00, 4.69},  {10.00, 5.31},
        {13.00, 5.50},  {13.00, 4.50}, {10.00, 4.00}, {10.00, 0.00},  {0.00, 0.00},   {0.00, 10.00},
        {10.00, 10.00}, {10.00, 6.00}, {15.00, 6.00}, {15.00, 4.00},  {7.00, 4.00},   {7.00, 3.00},
        {3.00, 3.00},   {3.00, 7.00},  {7.00, 7.00},  {7.00, 6.00},   {4.00, 6.00},   {4.00, 4.00},
        {0.00, 0.40},   {0.00, 0.00},  {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00},  {0.00, 0.60},
        {-0.50, 0.60},  {-0.50, 0.40}, {3.63, 0.00},  {10.00, 0.00},  {10.00, 10.00}, {0.00, 10.00},
        {0.00, 6.75},   {-4.60, 6.45}, {-5.12, 4.59}, {0.42, -5.63},  {2.11, 0.00},   {0.31, 0.00},
        {0.00, -3.00},  {0.00, 3.78},  {0.00, 1.74},  {-2.94, 2.27},  {-3.32, 4.11}};

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
    // WARNING
    // TODO WARNING - The correct results are for the algorithm WITHOUT considering the non-intersecting
    // polygons/holes - AFTER ADDING THIS FEATURE THE TEST MIGHT BREAK
    // The expected results of this test need to be MODIFIED after the correction is added

    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    polyTest<PolygonFunctions::polyUnion>(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst,
                                              polyBpolyIdxConst, polyBpointsIdxConst, polyBpolyPointsConst,
                                              outPolyIdx, outPointIdx, outPolyPoints, true, true, 1);


    std::vector<int32_t> outPolyIdxCorrect = {3};
    std::vector<int32_t> outPointIdxCorrect = {8, 11, 15};
    std::vector<NativeGeoPoint> outPolyPointsCorrect = {
        {3.63, 0.00},  {10.00, 0.00}, {10.00, 10.00}, {0.00, 10.00}, {0.00, 6.75},
        {-4.60, 6.45}, {-5.12, 4.59}, {0.42, -5.63},  {2.11, 0.00},  {0.31, 0.00},
        {0.00, -3.00}, {0.00, 3.78},  {0.00, 1.74},   {-2.94, 2.27}, {-3.32, 4.11}};

    // Check the results
    // polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);



	printPolygonAsGeoGebraPolygons(polyApolyIdxConst, polyApointsIdxConst, polyApolyPointsConst, 'A');
    printPolygonAsGeoGebraPolygons(polyBpolyIdxConst, polyBpointsIdxConst, polyBpolyPointsConst, 'F');
    printPolygonAsGeoGebraPolygons(outPolyIdx, outPointIdx, outPolyPoints, 'H');


	polyCompare(outPolyIdx, outPointIdx, outPolyPoints, outPolyIdxCorrect, outPointIdxCorrect, outPolyPointsCorrect);


    FAIL();
}


TEST(GPUPolygonClippingTests, NoIntersectionBInAOverlapTest)
{
    int32_t dataElementCount = 1;

    std::vector<int32_t> polyApolyIdx = {1};
    std::vector<int32_t> polyApointsIdx = {4};
    std::vector<NativeGeoPoint> polyApolyPoints = {{0.00, 0.00}, {2.00, 0.00}, {2.00, 2.00}, {0.00, 2.00}};

    std::vector<int32_t> polyBpolyIdx = {1};
    std::vector<int32_t> polyBpointsIdx = {4};
    std::vector<NativeGeoPoint> polyBpolyPoints = {{0.50, 0.50}, {1.50, 0.50}, {1.50, 1.50}, {0.50, 1.50}};

    std::vector<int32_t> outPolyIdx;
    std::vector<int32_t> outPointIdx;
    std::vector<NativeGeoPoint> outPolyPoints;

    // Run the intersect test
    /*
    polyTest<PolygonFunctions::polyIntersect>(polyApolyIdx, polyApointsIdx, polyApolyPoints, polyBpolyIdx,
                                              polyBpointsIdx, polyBpolyPoints, outPolyIdx, outPointIdx,
                                              outPolyPoints, false, false, dataElementCount);
											  */

    //printPolygonAsGeoGebraPolygons(polyApolyIdx, polyApointsIdx, polyApolyPoints, 'A');
    //printPolygonAsGeoGebraPolygons(polyBpolyIdx, polyBpointsIdx, polyBpolyPoints, 'B');
    //printPolygonAsGeoGebraPolygons(outPolyIdx, outPointIdx, outPolyPoints, 'C');

    // printPolygonAsList(outPolyIdx, outPointIdx, outPolyPoints, true);

    FAIL();
}
