#include <cstdint>
#include <cstdlib>
#include <memory>

#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../dropdbase/QueryEngine/GPUCore/cuda_ptr.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "gtest/gtest.h"

template <typename M, typename T>
void TestGenerateIndexes(std::vector<M> mask, std::vector<T> correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = mask.size();
    cuda_ptr<M> maskDevice(ELEMENT_COUNT);
    std::unique_ptr<T[]> resultHost = std::make_unique<T[]>(ELEMENT_COUNT);
    int32_t outCount;

    GPUMemory::copyHostToDevice(maskDevice.get(), mask.data(), ELEMENT_COUNT);

    GPUReconstruct::GenerateIndexes(resultHost.get(), &outCount, maskDevice.get(), ELEMENT_COUNT);

    ASSERT_EQ(outCount, correctOutput.size()) << "incorrect output count";
    for (int i = 0; i < outCount; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput[i]) << "at [" << i << "], mask: " << mask[i];
    }
}

template <typename M, typename T>
void TestGenerateIndexesKeep(std::vector<M> mask, std::vector<T> correctOutput)
{
    // Initialize CUDA context
    Context::getInstance();

    const int32_t ELEMENT_COUNT = mask.size();
    cuda_ptr<M> maskDevice(ELEMENT_COUNT);
    T* resultDevice;
    int32_t outCount;

    GPUMemory::copyHostToDevice(maskDevice.get(), mask.data(), ELEMENT_COUNT);

    GPUReconstruct::GenerateIndexesKeep(&resultDevice, &outCount, maskDevice.get(), ELEMENT_COUNT);

    std::unique_ptr<T[]> resultHost = std::make_unique<T[]>(outCount);
    GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice, outCount);
    GPUMemory::free(resultDevice);

    ASSERT_EQ(outCount, correctOutput.size()) << "incorrect output count";
    for (int i = 0; i < outCount; i++)
    {
        ASSERT_EQ(resultHost.get()[i], correctOutput[i]) << "at [" << i << "], mask: " << mask[i];
    }
}

void TestReconstructPolyCol(std::vector<int8_t> mask, std::vector<std::string> polygonsWkt,
	std::vector<std::vector<std::vector<NativeGeoPoint>>> correctPoints)
{
	// Initialize CUDA context
	Context::getInstance();
	const int32_t ELEMENT_COUNT = mask.size();
	cuda_ptr<int8_t> maskDevice(ELEMENT_COUNT);
	int32_t outCount;
	GPUMemory::copyHostToDevice(maskDevice.get(), mask.data(), ELEMENT_COUNT);

	// Convert WKT to protobuf format
	std::vector<ColmnarDB::Types::ComplexPolygon> protobufVector;
	for (auto& wkt : polygonsWkt)
	{
		protobufVector.emplace_back(ComplexPolygonFactory::FromWkt(wkt));
	}

	// Convert protobuf format to GPUPolygon
	GPUMemory::GPUPolygon polygonColumn = ComplexPolygonFactory::PrepareGPUPolygon(protobufVector);
	GPUMemory::GPUPolygon outCol;

	// Do the reconstruction
	GPUReconstruct::ReconstructPolyColKeep(&outCol, &outCount, polygonColumn, maskDevice.get(), ELEMENT_COUNT);

	// Check results
	ASSERT_EQ(outCount, correctPoints.size());
	if (outCount > 0)
	{
		// Copy results
		std::unique_ptr<int32_t[]> hostPolyIdx = std::make_unique<int32_t[]>(outCount);
		GPUMemory::copyDeviceToHost(hostPolyIdx.get(), outCol.polyIdx, outCount);
		int32_t subpolyCounts = hostPolyIdx[outCount - 1];

		std::unique_ptr<int32_t[]> hostPointIdx = std::make_unique<int32_t[]>(subpolyCounts);
		GPUMemory::copyDeviceToHost(hostPointIdx.get(), outCol.pointIdx, subpolyCounts);
		int32_t pointCounts = hostPointIdx[subpolyCounts - 1];

		std::unique_ptr<NativeGeoPoint[]> hostPoints = std::make_unique<NativeGeoPoint[]>(pointCounts);
		GPUMemory::copyDeviceToHost(hostPoints.get(), outCol.polyPoints, pointCounts);

		int32_t correctPolyIdx = 0;
		int32_t correctPointIdx = 0;
		for (int32_t i = 0; i < outCount; i++)	// for complex polygons
		{
			int32_t polyIdx_prev = (i == 0) ? 0 : hostPolyIdx[i - 1];

			ASSERT_EQ(polyIdx_prev, correctPolyIdx) << "polyIdx at " << i;
			correctPolyIdx += correctPoints[i].size();

			int32_t hostPolyCount = hostPolyIdx[i] - polyIdx_prev;
			ASSERT_EQ(correctPoints[i].size(), hostPolyCount);

			for (int32_t j = 0; j < correctPoints[i].size(); j++)	// for subpolygons
			{
				int32_t pointIdx_prev = (j == 0) ? ((i == 0) ? 0 : hostPointIdx[polyIdx_prev - 1]) : hostPointIdx[polyIdx_prev + j - 1];
				
				ASSERT_EQ(pointIdx_prev, correctPointIdx) << "pointIdx at " << i << "," << j;
				correctPointIdx += correctPoints[i][j].size();

				int32_t hostPointCount = hostPointIdx[polyIdx_prev + j] - pointIdx_prev;
				ASSERT_EQ(correctPoints[i][j].size(), hostPointCount);

				for (int32_t k = 0; k < correctPoints[i][j].size(); k++)	// for points
				{
					ASSERT_EQ(hostPoints[pointIdx_prev + k].latitude, correctPoints[i][j][k].latitude) << "polyPoint.latitude at " << i << "," << j << "," << k;
					ASSERT_EQ(hostPoints[pointIdx_prev + k].longitude, correctPoints[i][j][k].longitude) << "polyPoint.longitude at " << i << "," << j << "," << k;
				}
			}
		}
	}
	else
	{
		// For empty result check if all pointers in the GPUPolygon struct are nullptr
		ASSERT_EQ(outCol.polyPoints, nullptr);
		ASSERT_EQ(outCol.pointIdx, nullptr);
		ASSERT_EQ(outCol.polyIdx, nullptr);
	}
}


TEST(GPUReconstructTests, GenerateIndexesFullMask)
{
    TestGenerateIndexes(std::vector<int8_t>{1, 1, 1, 1}, std::vector<int32_t>{0, 1, 2, 3});
}

TEST(GPUReconstructTests, GenerateIndexesEmptyMask)
{
    TestGenerateIndexes(std::vector<int8_t>{0, 0, 0, 0, 0, 0}, std::vector<int32_t>{});
}

TEST(GPUReconstructTests, GenerateIndexesMixMask)
{
    TestGenerateIndexes(std::vector<int8_t>{0, 1, 0, 0, 1, 1, 1, 1}, std::vector<int32_t>{1, 4, 5, 6, 7});
}

TEST(GPUReconstructTests, GenerateIndexesBigMask)
{
    std::vector<int8_t> mask;
    std::vector<int32_t> correctOutput;
    for (int32_t i = 0; i < (1 << 18); i++)
    {
        int8_t maskTrue = i % 2;
        mask.emplace_back(maskTrue);
        if (maskTrue)
        {
            correctOutput.emplace_back(i);
        }
    }
    TestGenerateIndexes(mask, correctOutput);
}

TEST(GPUReconstructTests, GenerateIndexesKeepMixMask)
{
    TestGenerateIndexesKeep(std::vector<int8_t>{0, 1, 0, 0, 1, 1, 1, 1}, std::vector<int32_t>{1, 4, 5, 6, 7});
}

TEST(GPUReconstructTests, ReconstructPolyColHalfMask)
{
	TestReconstructPolyCol({ 0, 1 }, {
		"POLYGON((0 0, 1 1, 0 1, 0 0), (0 0, 1 1, 0 1, 0 0), (0 0, 1 2, 2 2, 3 1, 0 1, 0 0), (0 0, 1 0, 1 1, 0 1, 0 0))",
		"POLYGON((0 0, 1 1, 0 1, 0 0), (2 2, 3 2, 2 3, 2 2), (1.5 1.0, 0.0 0.0, 0.5 0.2, 1.5 1.0))"},
		{
			{
				{{0, 0}, {1, 1}, {0, 1}},
				{{2, 2}, {3, 2}, {2, 3}},
				{{1.5f, 1.0f}, {0.0f, 0.0f}, {0.5f, 0.2f}}
			}
		});
}


TEST(GPUReconstructTests, ReconstructPolyColFullMask)
{
	TestReconstructPolyCol({ 1, 1 }, {
		"POLYGON((0 2, 3 7, 1 1, 0 1, 0 2), (-1 -1, 2 1, 0 0, -1 -1))",
		"POLYGON((0 0, 1 1, 0 1, 0 0), (2 2, 3 2, 2 3, 2 2))" },
		{
			{
				{{0, 2}, {3, 7}, {1, 1}, {0, 1}},
				{{-1, -1}, {2, 1}, {0, 0}}
			},
			{
				{{0, 0}, {1, 1}, {0, 1}},
				{{2, 2}, {3, 2}, {2, 3}}
			}
		});
}

TEST(GPUReconstructTests, ReconstructPolyColEmptyMask)
{
	TestReconstructPolyCol({ 0, 0 }, {
		"POLYGON((0 2, 3 7, 1 1, 0 1, 0 2), (-1 -1, 2 1, 0 0, -1 -1))",
		"POLYGON((0 0, 1 1, 0 1, 0 0), (2 2, 3 2, 2 3, 2 2))" },
		{});
}
