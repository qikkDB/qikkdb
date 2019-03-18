#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUArithmetic.cuh"
#include "GPUMemory.cuh"

#include "../../../cub/cub.cuh"
#include "../../NativeGeoPoint.h"
#include "../Context.h"

// WARNING - This version clips only the 0th polygon of a complex polygon !!!

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
    bool is_intersect;

    int32_t nextIdx;
    int32_t prevIdx;
    int32_t cross_link; // Cross ink also indicates if a node is a intersect or not = only intersects have positive cross links
};

// Vertex counts in each input polygon - needed for offset calculation
__device__ int32_t* poly1VertexCounts;
__device__ int32_t* poly2VertexCounts;

// Vertex count start offsets
//__device__ int32_t* poly1VertexOffsets;
//__device__ int32_t* poly2VertexOffsets;

// Doubly linked list offset buffers
__device__ int32_t* DLLVertexCounts;
__device__ int32_t* DLLVertexCountOffsets;

//__device__ int32_t* DLLPolygonCounts;
//__device__ int32_t* DLLPolygonCountOffsets;

// Data buffers for doubly linked lists of polygons during clipping
__device__ PolygonNodeDLL* poly1DLList;
__device__ PolygonNodeDLL* poly2DLList;

// Offset buffer sizes - TODO make device callable
// static int32_t poly1VertexCountTotal;
// static int32_t poly2VertexCountTotal;

// Total/Maximal size of the result DLL list vertices and polygons
static int32_t DLLVertexCountTotal;
// static int32_t DLLPolygonCountTotal;

// A kernel for counting the number of vertices that a complex polygon has
inline __global__ void kernel_calculate_point_count_in_complex_polygon(int32_t* pointCounts,
                                                                       GPUMemory::GPUPolygon complexPolygon,
                                                                       int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        /*
        // Sum the number of points of a polygon and store it in a buffer
        int32_t vertexCountSum = 0;
        for (int32_t j = 0; j < complexPolygon.polyCount[i]; j++)
        {
            vertexCountSum += complexPolygon.pointCount[complexPolygon.polyIdx[i] + j];
        }
        pointCounts[i] = vertexCountSum;
        */

        // Account only for the 0th polygon in a complex polygon - the 0 is only for better understanding
        pointCounts[i] = complexPolygon.pointCount[complexPolygon.polyIdx[i] + 0];
    }
}

template <typename OP>
__global__ void kernel_polygon_clipping(GPUMemory::GPUPolygon complexPolygonOut,
                                        GPUMemory::GPUPolygon complexPolygon1,
                                        GPUMemory::GPUPolygon complexPolygon2,
                                        int32_t dataElementCount)
{
    const int32_t ROOT_NODE_IDX = 0;

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Transform - update the input polygons to Doubly Linked Lists (DLLs)
        // Iterate over the vertices of the first poly of the complex poly
        // Only for 0th polygon

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Poly 1
        // Pointer to the end of the list ( one element after the last element)
        int32_t DLLPoly1ElementCount = 0;
        for (int32_t j = 0; j < complexPolygon1.pointCount[complexPolygon1.polyIdx[i] + 0]; j++)
        {
            // Get the coordinates
            float x = complexPolygon1
                          .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] + j]
                          .latitude;
            float y = complexPolygon1
                          .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] + j]
                          .longitude;

            // Create an empty node
            PolygonNodeDLL tempNode;

            tempNode.poly_group = -1;
            tempNode.point = {x, y};

            tempNode.linear_distance = 0;
            tempNode.is_intersect = false;

            tempNode.cross_link = -1;

            // Get the offset index - needed because of the character of the prefix sum
            int32_t DLLVertexCountOffsetIdx = 0;
            if ((i - 1) < 0)
            {
                DLLVertexCountOffsetIdx = 0;
            }
            else
            {
                DLLVertexCountOffsetIdx = DLLVertexCountOffsets[i - 1];
            }

            // Rewire the "pointers"
            if (DLLPoly1ElementCount == ROOT_NODE_IDX)
            {
                // Set the "pointers"
                // root just points to itself:
                //    +-> (root) <-+
                //    |            |
                //    +------------+
                tempNode.nextIdx = ROOT_NODE_IDX;
                tempNode.prevIdx = ROOT_NODE_IDX;
            }
            else
            {
                // change this:
                //    ...-- (prev) <--------------> (root) --...
                // to this:
                //    ...-- (prev) <--> (node) <--> (root) --...

                int32_t oldLastElementIdx = poly1DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].prevIdx;
                poly1DLList[DLLVertexCountOffsetIdx + oldLastElementIdx].nextIdx = DLLPoly1ElementCount;
                tempNode.prevIdx = oldLastElementIdx;
                tempNode.nextIdx = ROOT_NODE_IDX;
                poly1DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].prevIdx = DLLPoly1ElementCount;
            }

            // Insert the node into the list
            poly1DLList[DLLVertexCountOffsetIdx + DLLPoly1ElementCount] = tempNode;

            // Increment the number of elements
            DLLPoly1ElementCount++;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Poly 2
        // Pointer to the end of the list ( one element after the last element)
        int32_t DLLPoly2ElementCount = 0;
        for (int32_t j = 0; j < complexPolygon2.pointCount[complexPolygon2.polyIdx[i] + 0]; j++)
        {
            // Get the coordinates
            float x = complexPolygon2
                          .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] + j]
                          .latitude;
            float y = complexPolygon2
                          .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] + j]
                          .longitude;

            // Create an empty node
            PolygonNodeDLL tempNode;

            tempNode.poly_group = -1;
            tempNode.point = {x, y};

            tempNode.linear_distance = 0;
            tempNode.is_intersect = false;

            tempNode.cross_link = -1;

            // Get the offset index - needed because of the character of the prefix sum
            int32_t DLLVertexCountOffsetIdx = 0;
            if ((i - 1) < 0)
            {
                DLLVertexCountOffsetIdx = 0;
            }
            else
            {
                DLLVertexCountOffsetIdx = DLLVertexCountOffsets[i - 1];
            }

            // Rewire the "pointers"
            if (DLLPoly2ElementCount == ROOT_NODE_IDX)
            {
                // Set the "pointers"
                // root just points to itself:
                //    +-> (root) <-+
                //    |            |
                //    +------------+
                tempNode.nextIdx = ROOT_NODE_IDX;
                tempNode.prevIdx = ROOT_NODE_IDX;
            }
            else
            {
                // change this:
                //    ...-- (prev) <--------------> (root) --...
                // to this:
                //    ...-- (prev) <--> (node) <--> (root) --...

                int32_t oldLastElementIdx = poly2DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].prevIdx;
                poly2DLList[DLLVertexCountOffsetIdx + oldLastElementIdx].nextIdx = DLLPoly2ElementCount;
                tempNode.prevIdx = oldLastElementIdx;
                tempNode.nextIdx = ROOT_NODE_IDX;
                poly2DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].prevIdx = DLLPoly2ElementCount;
            }

            // Insert the node into the list
            poly2DLList[DLLVertexCountOffsetIdx + DLLPoly2ElementCount] = tempNode;

            // Increment the number of elements
            DLLPoly2ElementCount++;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate all line intersections and append them to the polygon DLLs

		// TODO Change to DLLs reading
        for (int32_t j = 0; j < complexPolygon1.pointCount[complexPolygon1.polyIdx[i] + 0]; j++)
        {
            // Get the start vertex s and the end vertex e of the first edge
            // The end vertex needs the modulus to wrap around to the first vertex
            float s1_x = complexPolygon1
                             .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] + j]
                             .latitude;
            float s1_y = complexPolygon1
                             .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] + j]
                             .longitude;

            float e1_x =
                complexPolygon1
                    .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] +
                                (j + 1) % complexPolygon1.pointCount[complexPolygon1.polyIdx[i] + 0]]
                    .latitude;
            float e1_y =
                complexPolygon1
                    .polyPoints[complexPolygon1.pointIdx[complexPolygon1.polyIdx[i] + 0] +
                                (j + 1) % complexPolygon1.pointCount[complexPolygon1.polyIdx[i] + 0]]
                    .longitude;


            for (int32_t k = 0; k < complexPolygon2.pointCount[complexPolygon2.polyIdx[i] + 0]; j++)
            {
                // Get the start vertex s and the end vertex e of the second edge
                // The end vertex needs the modulus to wrap around to the first vertex
                float s2_x = complexPolygon2
                                 .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] + k]
                                 .latitude;
                float s2_y = complexPolygon2
                                 .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] + k]
                                 .longitude;

                float e2_x =
                    complexPolygon2
                        .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] +
                                    (k + 1) % complexPolygon2.pointCount[complexPolygon2.polyIdx[i] + 0]]
                        .latitude;
                float e2_y =
                    complexPolygon2
                        .polyPoints[complexPolygon2.pointIdx[complexPolygon2.polyIdx[i] + 0] +
                                    (k + 1) % complexPolygon2.pointCount[complexPolygon2.polyIdx[i] + 0]]
                        .longitude;

				//////////////////////////////////////////////////////////////////////////////////////////
				// Calculate the intersections - create a new node, push it to both lists based on linear distance and cross reference it

            }
        }
    }
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
        GPUMemory::allocAndSet(&poly1VertexCounts, 0, dataElementCount);
        GPUMemory::allocAndSet(&poly2VertexCounts, 0, dataElementCount);

        // Calculate the number of vertices in the complex polygons
        kernel_calculate_point_count_in_complex_polygon<<<context.calcGridDim(dataElementCount),
                                                          context.getBlockDim()>>>(poly1VertexCounts, polygon1,
                                                                                   dataElementCount);
        kernel_calculate_point_count_in_complex_polygon<<<context.calcGridDim(dataElementCount),
                                                          context.getBlockDim()>>>(poly2VertexCounts, polygon2,
                                                                                   dataElementCount);
        /*
        // Alloc the offset buffs
        GPUMemory::allocAndSet(&poly1VertexOffsets, 0, dataElementCount);
        GPUMemory::allocAndSet(&poly2VertexOffsets, 0, dataElementCount);

        // Transform the offset buffers using the exclusive prefix sum - inclusive sum with 0 as the
        0th element - think about it again
        // Input poly 1 offsets - prefix sum calculation
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly1VertexCounts,
        poly1VertexOffsets, dataElementCount); GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage),
        temp_storage_bytes); cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
        poly1VertexCounts, poly1VertexOffsets, dataElementCount); GPUMemory::free(d_temp_storage);

        // Input poly 2 offsets - prefix sum calculation
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, poly2VertexCounts,
        poly2VertexOffsets, dataElementCount); GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage),
        temp_storage_bytes); cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
        poly2VertexCounts, poly2VertexOffsets, dataElementCount); GPUMemory::free(d_temp_storage);

        //Copy back the last element of both prefix sum calculations - the total number of points
        poly1VertexCountTotal = 0;
        poly2VertexCountTotal = 0;

        GPUMemory::copyDeviceToHost(&poly1VertexCountTotal, poly1VertexOffsets + dataElementCount -
        1, 1); GPUMemory::copyDeviceToHost(&poly2VertexCountTotal, poly2VertexOffsets +
        dataElementCount - 1, 1);
        */

        /*
        // TODO REMOVE Debug code - copy back buffer 1 to see it's content
        int32_t offsets1[2];
        int32_t offsets2[2];
        GPUMemory::copyDeviceToHost(offsets1, poly1VertexCounts, dataElementCount);
        GPUMemory::copyDeviceToHost(offsets2, poly2VertexCounts, dataElementCount);
        */

        ///////////////////////////////////////////////////////////////////////////////////////
        // Pre allocte the output buffer based on the formula n*k + n + k (one polygon only)
        // The points of polygonOut structure has to be (n*k + n + k) times dataElementCount ( for each polygon) items wide
        // The result is saved as a list of ComplexPolygons
        // The number of polygons is max (n*k + n + k)/3
        // The prefix sum is also needed for output indexing
        // Now calculate the result sizes and alloate the outPoly structure

        // DLL VERTEX SPACE CALCULATION AND ALOCATION
        // Calc maximum needed size - real values will be smaller due to assumption of the worst case
        DLLVertexCounts = nullptr;
        GPUMemory::allocAndSet(&DLLVertexCounts, 0, dataElementCount);

        // Calc (n*k + n + k) for each input row
        GPUArithmetic::colCol<ArithmeticOperations::mul>(DLLVertexCounts, poly1VertexCounts,
                                                         poly2VertexCounts, dataElementCount);
        GPUArithmetic::colCol<ArithmeticOperations::add>(DLLVertexCounts, DLLVertexCounts,
                                                         poly1VertexCounts, dataElementCount);
        GPUArithmetic::colCol<ArithmeticOperations::add>(DLLVertexCounts, DLLVertexCounts,
                                                         poly2VertexCounts, dataElementCount);

        // Calc the prefix sum for each input row - points
        DLLVertexCountOffsets = nullptr;
        GPUMemory::allocAndSet(&DLLVertexCountOffsets, 0, dataElementCount);

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLVertexCounts,
                                      DLLVertexCountOffsets, dataElementCount);
        GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLVertexCounts,
                                      DLLVertexCountOffsets, dataElementCount);
        GPUMemory::free(d_temp_storage);

        // Get the number of maximum point count for the result buffer allocation
        DLLVertexCountTotal = 0;

        GPUMemory::copyDeviceToHost(&DLLVertexCountTotal, DLLVertexCountOffsets + dataElementCount - 1, 1);

        // DLL POLYGON SPACE CALCULATION AND ALLOCATION
        // TODO think about geting rid of this - only size needed, possible to calculate at runtime
        // Because a/3 + b/3 + c/3 == (a + b + c)/3
        /*
        DLLPolygonCounts = nullptr;
        GPUMemory::allocAndSet(&DLLPolygonCounts, 0, dataElementCount);

        // Calc (n*k + n + k)/3 for each input row
        GPUArithmetic::colConst<ArithmeticOperations::div>
            (DLLPolygonCounts, DLLVertexCounts, 3, dataElementCount);

        // Calc the prefix sum for each input row - polygons
        DLLPolygonCountOffsets = nullptr;
        GPUMemory::allocAndSet(&DLLPolygonCountOffsets, 0, dataElementCount);

        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLPolygonCounts, DLLPolygonCountOffsets, dataElementCount);
        GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLPolygonCounts, DLLPolygonCountOffsets, dataElementCount);
        GPUMemory::free(d_temp_storage);

        // Get the number of maximum polygon count for the result buffer allocation
        DLLPolygonCountTotal = 0;

        GPUMemory::copyDeviceToHost(&DLLPolygonCountTotal, DLLPolygonCountOffsets + dataElementCount - 1, 1);
        */

        ///////////////////////////////////////////////////////////////////////////////////////
        // Alloc the buffer for doubly linked lists - create a temporary out polygon
        GPUMemory::GPUPolygon polygonOutTemp;

        // The data buffer for linked lists - the max count of the vertices is the result of the
        // prefix sum Alloc space for the doubly linked lists for both polygons in both collumns
        GPUMemory::alloc(&poly1DLList, DLLVertexCountTotal);
        GPUMemory::alloc(&poly2DLList, DLLVertexCountTotal);

        ///////////////////////////////////////////////////////////////////////////////////////
        // Allocation of the result buffer - worst case scenario
        // The total count of resulting vertices
        GPUMemory::alloc(&polygonOutTemp.polyPoints, DLLVertexCountTotal);

        // The number of complex polygons is the same as the input complex polygons
        GPUMemory::alloc(&polygonOutTemp.polyIdx, dataElementCount);
        GPUMemory::alloc(&polygonOutTemp.polyCount, dataElementCount);

        // The number of simple polygons is (n*k + n + k)/3 summed over all polygons
        // GPUMemory::alloc(&polygonOutTemp.pointIdx, DLLPolygonCountTotal);
        // GPUMemory::alloc(&polygonOutTemp.pointCount, DLLPolygonCountTotal);
        GPUMemory::alloc(&polygonOutTemp.pointIdx, DLLVertexCountTotal / 3);
        GPUMemory::alloc(&polygonOutTemp.pointCount, DLLVertexCountTotal / 3);

        // Run the clipping kernel
        kernel_polygon_clipping<OP>
            <<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(polygonOutTemp, polygon1,
                                                                               polygon2, dataElementCount);

        // TODO Reconstruct the real output polygon from temp output polygon

        // Free the tempora polygon
        GPUMemory::free(polygonOutTemp.polyPoints);
        GPUMemory::free(polygonOutTemp.polyIdx);
        GPUMemory::free(polygonOutTemp.polyCount);
        GPUMemory::free(polygonOutTemp.pointIdx);
        GPUMemory::free(polygonOutTemp.pointCount);

        // Free the allocated memory
        GPUMemory::free(poly1VertexCounts);
        GPUMemory::free(poly2VertexCounts);

        // GPUMemory::free(poly1VertexOffsets);
        // GPUMemory::free(poly2VertexOffsets);

        GPUMemory::free(poly1DLList);
        GPUMemory::free(poly2DLList);

        GPUMemory::free(DLLVertexCounts);
        GPUMemory::free(DLLVertexCountOffsets);
        // GPUMemory::free(DLLPolygonCounts);
        // GPUMemory::free(DLLPolygonCountOffsets);

        // Set error
        QueryEngineError::setCudaError(cudaGetLastError());
    }
};
