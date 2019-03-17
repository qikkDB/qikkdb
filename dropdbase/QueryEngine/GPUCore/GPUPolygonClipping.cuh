#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMemory.cuh"
#include "GPUArithmetic.cuh"

#include "../../NativeGeoPoint.h"
#include "../Context.h"
#include "../../../cub/cub.cuh"

namespace PolygonFunctions
{
struct polyIntersect
{
    __device__ __host__ void operator()() const
    {
    }
};

struct polyUnion
{
    __device__ __host__ void operator()() const
    {
    }
};
} // namespace PolygonFunctions

// Struct for the polygon Doubly Linked List construction on the GPU
__device__ struct PolygonNodeDLL
{
    int32_t poly_group;

    NativeGeoPoint point;

    float linear_distance;
    int32_t is_intersect;

    int32_t next;
    int32_t prev;
    int32_t cross_link;
};

// Data buffers for doubly linked lists of polygons during clipping
__device__ PolygonNodeDLL* poly1DLList;
__device__ PolygonNodeDLL* poly2DLList;

// Doubly linked list vertex counts
__device__ int32_t* poly1DLListVertexCounts;
__device__ int32_t* poly2DLListVertexCounts;

// Doubly linked list starts offset
__device__ int32_t* poly1DLListOffset;
__device__ int32_t* poly2DLListOffset;

// Offset buffers
__device__ int32_t* resultMaximumVertexCounts;
__device__ int32_t* resultMaximumVertexCountOffset;
__device__ int32_t* resultMaximumPolygonCounts;
__device__ int32_t* resultMaximumPolygonCountOffset;

//Offset buffer sizes - TODO make device callable
static int32_t pointCountTotalPoly1;
static int32_t pointCountTotalPoly2;

static int32_t resultMaximumVertexCountTotal;
static int32_t resultMaximumPolygonCountTotal;

// A kernel for counting the number of vertices that a complex polygon has
inline __global__ void kernel_calculate_points_in_complex_polygon_count(int32_t* pointCounts,
                                                             GPUMemory::GPUPolygon complexPolygon,
                                                             int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Sum the number of points of a polygon and store it in a buffer
        int32_t vertexCountSum = 0;
        for (int32_t j = 0; j < complexPolygon.polyCount[i]; j++)
        {
            vertexCountSum += complexPolygon.pointCount[complexPolygon.polyIdx[i] + j];
        }
        pointCounts[i] = vertexCountSum;
    }
}

template <typename OP>
__global__ void kernel_polygon_clipping(GPUMemory::GPUPolygon complexPolygonOut,
                                        GPUMemory::GPUPolygon complexPolygon1,
                                        GPUMemory::GPUPolygon complexPolygon2,
                                        int32_t dataElementCount)
{
}

class GPUPolygonClip
{
public:
    template <typename OP>
    static void
    ColCol(GPUMemory::GPUPolygon polygonOut, GPUMemory::GPUPolygon polygon1, GPUMemory::GPUPolygon polygon2, int32_t dataElementCount)
    {
        // Get the context instance
        Context& context = Context::getInstance();

        // Precalcualte the maximal needed size for a doubly linked list as
        // n*k + n + k where n is the number of vertices of polygon 1 and k is the number of
        // vertices of polygon 2 This is a case for one row - for all rows this has to be done
        // dataElementCount times Offsets for the linked list start indexes are needed too - these
        // can be calculated as the prefix sum of doubly linked list sizes The result size of the
        // doubly linked list buffer is a sum of all dll buffers

        // Alloc the vertex count buffs
        GPUMemory::allocAndSet(&poly1DLListVertexCounts, 0, dataElementCount);
        GPUMemory::allocAndSet(&poly2DLListVertexCounts, 0, dataElementCount);

        // Calculate the number of vertices in the complex polygons
        kernel_calculate_points_in_complex_polygon_count<<<context.calcGridDim(dataElementCount),
                                                       context.getBlockDim()>>>(poly1DLListVertexCounts, polygon1,
                                                                                dataElementCount);
        kernel_calculate_points_in_complex_polygon_count<<<context.calcGridDim(dataElementCount),
                                                       context.getBlockDim()>>>(poly2DLListVertexCounts, polygon2,
                                                                                dataElementCount);
		// Alloc the offset buffs
        GPUMemory::allocAndSet(&poly1DLListOffset, 0, dataElementCount);
        GPUMemory::allocAndSet(&poly2DLListOffset, 0, dataElementCount);

        // Transform the offset buffers using the exclusive prefix sum - inclusive sum with 0 as the 0th element - think about it again
		// Input poly 1 offsets - prefix sum calculation
		void *d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly1DLListVertexCounts, poly1DLListOffset, dataElementCount);
		GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly1DLListVertexCounts, poly1DLListOffset, dataElementCount);
		GPUMemory::free(d_temp_storage);

		// Input poly 2 offsets - prefix sum calculation
		d_temp_storage = nullptr;
		temp_storage_bytes = 0;
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly2DLListVertexCounts, poly2DLListOffset, dataElementCount);
		GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly2DLListVertexCounts, poly2DLListOffset, dataElementCount);
		GPUMemory::free(d_temp_storage);

		//Copy back the last element of both prefix sum calculations - the total number of points
		pointCountTotalPoly1 = 0;
		pointCountTotalPoly2 = 0;

		GPUMemory::copyDeviceToHost(&pointCountTotalPoly1, poly1DLListOffset + dataElementCount - 1, 1);
		GPUMemory::copyDeviceToHost(&pointCountTotalPoly2, poly2DLListOffset + dataElementCount - 1, 1);

		/*
		// TODO REMOVE Debug code - copy back buffer 1 to see it's content
		int32_t offsets1[2];
		int32_t offsets2[2];
		GPUMemory::copyDeviceToHost(offsets1, poly1DLListOffset, dataElementCount);
		GPUMemory::copyDeviceToHost(offsets2, poly2DLListOffset, dataElementCount);
		*/	

        // The data sandbox for linked lists - the max count of the vertices is the result of the prefix sum
        // Alloc space for the doubly linked lists for both polygons in both collumns
        GPUMemory::alloc(&poly1DLList, pointCountTotalPoly1);
        GPUMemory::alloc(&poly2DLList, pointCountTotalPoly2);

		///////////////////////////////////////////////////////////////////////////////////////	
		// Pre allocte the output buffer based on the formula n*k + n + k (one polygon only)
		// The points of polygonOut structure has to be (n*k + n + k) times dataElementCount ( for each polygon) items wide
		// The result is saved as a list of ComplexPolygons
		// The number of polygons is max (n*k + n + k)/3
		// The prefix sum is also needed for output indexing
		// Now calculate the result sizes and alloate the outPoly structure

		// POINT SPACE CALCULATION AND ALOCATION
		// Calc maximum needed size - real values will be smaller due to assumption of the worst case
		resultMaximumVertexCounts = nullptr;
		GPUMemory::allocAndSet(&resultMaximumVertexCounts, 0, dataElementCount);

		// Calc (n*k + n + k) for each input row
		GPUArithmetic::colCol<ArithmeticOperations::mul>
			(resultMaximumVertexCounts, poly1DLListVertexCounts, poly2DLListVertexCounts, dataElementCount);
		GPUArithmetic::colCol<ArithmeticOperations::add>
			(resultMaximumVertexCounts, resultMaximumVertexCounts, poly1DLListVertexCounts, dataElementCount);
		GPUArithmetic::colCol<ArithmeticOperations::add>
			(resultMaximumVertexCounts, resultMaximumVertexCounts, poly2DLListVertexCounts, dataElementCount);	

		// Calc the prefix sum for each input row - points
		resultMaximumVertexCountOffset = nullptr;
		GPUMemory::allocAndSet(&resultMaximumVertexCountOffset, 0, dataElementCount);

		d_temp_storage = nullptr;
		temp_storage_bytes = 0;
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, resultMaximumVertexCounts, resultMaximumVertexCountOffset, dataElementCount);
		GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, resultMaximumVertexCounts, resultMaximumVertexCountOffset, dataElementCount);
		GPUMemory::free(d_temp_storage);

		// Get the number of maximum point count for the result buffer allocation
		resultMaximumVertexCountTotal = 0;

		GPUMemory::copyDeviceToHost(&resultMaximumVertexCountTotal, resultMaximumVertexCountOffset + dataElementCount - 1, 1);
		
		// POLYGON OFFSET SPACE CALCULATION AND ALLOCATION
		resultMaximumPolygonCounts = nullptr;
		GPUMemory::allocAndSet(&resultMaximumPolygonCounts, 0, dataElementCount);

		// Calc (n*k + n + k)/3 for each input row
		GPUArithmetic::colConst<ArithmeticOperations::div>
			(resultMaximumPolygonCounts, resultMaximumVertexCounts, 3, dataElementCount);

		// Calc the prefix sum for each input row - polygons
		resultMaximumPolygonCountOffset = nullptr;
		GPUMemory::allocAndSet(&resultMaximumPolygonCountOffset, 0, dataElementCount);

		d_temp_storage = nullptr;
		temp_storage_bytes = 0;
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, resultMaximumPolygonCounts, resultMaximumPolygonCountOffset, dataElementCount);
		GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, resultMaximumPolygonCounts, resultMaximumPolygonCountOffset, dataElementCount);
		GPUMemory::free(d_temp_storage);

		// Get the number of maximum polygon count for the result buffer allocation
		resultMaximumPolygonCountTotal = 0;

		GPUMemory::copyDeviceToHost(&resultMaximumPolygonCountTotal, resultMaximumPolygonCountOffset +  dataElementCount - 1, 1);

		///////////////////////////////////////////////////////////////////////////////////////
		// Allocation of the result buffer - worst case scenario
		// The total count of resulting vertices
		GPUMemory::alloc(&polygonOut.polyPoints, resultMaximumVertexCountTotal);	

		// The number of complex polygons is the same as the input complex polygons
		GPUMemory::alloc(&polygonOut.polyIdx, dataElementCount);
		GPUMemory::alloc(&polygonOut.polyCount, dataElementCount);
		
		// The number of simple polygons is (n*k + n + k)/3 summed over all polygons
		GPUMemory::alloc(&polygonOut.pointIdx, resultMaximumPolygonCountTotal);
		GPUMemory::alloc(&polygonOut.pointCount, resultMaximumPolygonCountTotal);

		//Run the clipping kernel
        kernel_polygon_clipping<OP>
            <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(polygonOut, polygon1,
                                                                               polygon2, dataElementCount);
        
		// Free the allocated memory
        GPUMemory::free(poly1DLListVertexCounts);
        GPUMemory::free(poly2DLListVertexCounts);

		GPUMemory::free(poly1DLListOffset);
        GPUMemory::free(poly2DLListOffset);

		GPUMemory::free(poly1DLList);
		GPUMemory::free(poly2DLList);

		GPUMemory::free(resultMaximumVertexCounts);
		GPUMemory::free(resultMaximumVertexCountOffset);
		GPUMemory::free(resultMaximumPolygonCounts);
		GPUMemory::free(resultMaximumPolygonCountOffset);

		// Set error
		QueryEngineError::setCudaError(cudaGetLastError());
    }
};
