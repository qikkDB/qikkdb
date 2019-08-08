#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUPolygonClipping.cuh"

#include "gtest/gtest.h"

TEST(GPUPolygonClippingTests, PoygonTest)
{
	int32_t dataElementCount = 2;

	std::vector<int32_t> polyApolyIdx = {3, 4};
	std::vector<int32_t> polyApointsIdx = {4, 8, 12, 15};
	std::vector<NativeGeoPoint> polyApolyPoints = {
		{4.5, 5.5},
		{6.0, 5.5},
		{6.0, 4.5},
		{4.5, 4.5},
		{10.0, 0.0},
		{0.0, 0.0},
		{0.0, 10.0},
		{10.0, 10.0},
		{7.0, 7.0},
		{3.0, 7.0},
		{3.0, 3.0},
		{7.0, 3.0},

		{0.0, 0.0},
		{1.0, 0.0},
		{0.5, 1.0}
	};

	std::vector<int32_t> polyBpolyIdx = {2, 3};
	std::vector<int32_t> polyBpointsIdx = {3, 7, 11};
	std::vector<NativeGeoPoint> polyBpolyPoints = {
		{13.0, 5.5},
		{13.0, 4.5},
		{5.0, 5.0},
		{4.0, 4.0},
		{15.0, 4.0},
		{15.0, 6.0},
		{4.0, 6.0},

		{0.0, 0.4},
		{1.0, 0.4},
		{1.0, 0.6},
		{0.0, 0.6}
	};

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
	GPUMemory::GPUPolygon polygonOut;	// This needs to be empty
	GPUPolygonClipping::ColCol<PolygonFunctions::polyIntersect>(polygonOut, polygonA, polygonB, dataElementCount);

	// Free the polygons
	GPUMemory::free(polygonA);
	GPUMemory::free(polygonB);
	GPUMemory::free(polygonOut);

}