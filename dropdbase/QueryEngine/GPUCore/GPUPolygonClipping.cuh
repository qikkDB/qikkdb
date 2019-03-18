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
    bool is_entry;

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
    // The root of the DLL is always the 0th element
    const int32_t ROOT_NODE_IDX = 0;

    // "Infinity"
    const float INF = 1000000000;

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Transform the input polygons to Doubly Linked Lists (DLLs)
        // Only for 0th polygon

        // Get the offset index for the dynamic list index - needed because of the character of the prefix sum
        int32_t DLLVertexCountOffsetIdx = 0;
        if ((i - 1) < 0)
        {
            DLLVertexCountOffsetIdx = 0;
        }
        else
        {
            DLLVertexCountOffsetIdx = DLLVertexCountOffsets[i - 1];
        }

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
            tempNode.is_entry = false;

            tempNode.cross_link = -1;

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
            tempNode.is_entry = false;

            tempNode.cross_link = -1;

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
        // Calculate all line intersections and append them to the polygon DLLs dynamiclaly during calculation

        // The root index of both lists
        int32_t here1Idx = ROOT_NODE_IDX;
        int32_t here2Idx = ROOT_NODE_IDX;

        int32_t next1Idx = here1Idx;
        int32_t next2Idx = here2Idx;

        do
        {
            do
            {
                ///////////////////////////////////////////////////////////////////////////////
                // Test intersection between:
                // here1Idx -> NextNonIntersection(here1Idx)  and
                // here2Idx -> NextNonIntersection(here2Idx)

                // Find the next non intersection edge
                next1Idx = here1Idx;
                next2Idx = here2Idx;

                do
                {
                    next1Idx = poly1DLList[DLLVertexCountOffsetIdx + next1Idx].nextIdx;
                } while (poly1DLList[DLLVertexCountOffsetIdx + next1Idx].is_intersect != false);

                do
                {
                    next2Idx = poly2DLList[DLLVertexCountOffsetIdx + next2Idx].nextIdx;
                } while (poly2DLList[DLLVertexCountOffsetIdx + next2Idx].is_intersect != false);

                ///////////////////////////////////////////////////////////////////////////////
                // Calculate the intersect - math is complex - see doc
                float adx = poly1DLList[DLLVertexCountOffsetIdx + next1Idx].point.latitude -
                            poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.latitude;
                float ady = poly1DLList[DLLVertexCountOffsetIdx + next1Idx].point.longitude -
                            poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.longitude;
                float bdx = poly2DLList[DLLVertexCountOffsetIdx + next2Idx].point.latitude -
                            poly2DLList[DLLVertexCountOffsetIdx + here2Idx].point.latitude;
                float bdy = poly2DLList[DLLVertexCountOffsetIdx + next2Idx].point.longitude -
                            poly2DLList[DLLVertexCountOffsetIdx + here2Idx].point.longitude;

                float axb = adx * bdy - ady * bdx;

                float cross = axb;
                float alongA = INF;
                float alongB = INF;

                float point_x = INF;
                float point_y = INF;

                if (axb == 0)
                {
                    // TODO Do something when lines are parallel !!!
                }

                float dx = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.latitude -
                           poly2DLList[DLLVertexCountOffsetIdx + here2Idx].point.latitude;
                float dy = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.longitude -
                           poly2DLList[DLLVertexCountOffsetIdx + here2Idx].point.longitude;

                alongA = (bdx * dy - bdy * dx) / axb;
                alongB = (adx * dy - ady * dx) / axb;

                point_x = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.latitude + alongA * adx;
                point_y = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].point.longitude + alongA * ady;

                ///////////////////////////////////////////////////////////////////////////////
                // If there is an intersection - add a new intersection vertex
                if (alongA > 0 && alongA < 1 && alongB > 0 && alongB < 1)
                {
                    // Insert intersection points in both polygons at the correct location, referencing each other Create an empty node
                    // Create the intersect node in firstlist
                    PolygonNodeDLL tempNodeA;

                    tempNodeA.poly_group = -1;
                    tempNodeA.point = {point_x, point_y};

                    tempNodeA.linear_distance = alongA;
                    tempNodeA.is_intersect = true;
                    tempNodeA.is_entry = false;

                    tempNodeA.cross_link = DLLPoly2ElementCount;

                    // Create the intersect node in second list
                    PolygonNodeDLL tempNodeB;

                    tempNodeB.poly_group = -1;
                    tempNodeB.point = {point_x, point_y};

                    tempNodeB.linear_distance = alongB;
                    tempNodeB.is_intersect = true;
                    tempNodeB.is_entry = false;

                    tempNodeB.cross_link = DLLPoly1ElementCount;

                    //////////////////////////////////////////////////////////////
                    // Find insertion between here1Idx and next1Idx, based on dist
                    int32_t inextIdx, iprevIdx;

                    inextIdx = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].nextIdx;
                    while (inextIdx != next1Idx && poly1DLList[DLLVertexCountOffsetIdx + inextIdx].linear_distance <
                                                       tempNodeA.linear_distance)
                    {
                        inextIdx = poly1DLList[DLLVertexCountOffsetIdx + inextIdx].nextIdx;
                    }
                    iprevIdx = poly1DLList[DLLVertexCountOffsetIdx + inextIdx].prevIdx;

                    // Insert node1 between iprev and inext
                    poly1DLList[DLLVertexCountOffsetIdx + inextIdx].prevIdx = DLLPoly1ElementCount;
                    tempNodeA.nextIdx = inextIdx;
                    tempNodeA.prevIdx = iprevIdx;
                    poly1DLList[DLLVertexCountOffsetIdx + iprevIdx].nextIdx = DLLPoly1ElementCount;

                    //////////////////////////////////////////////////////////////
                    // Find insertion between here2Idx and next2Idx, based on dist
                    inextIdx = poly2DLList[DLLVertexCountOffsetIdx + here2Idx].nextIdx;
                    while (inextIdx != next2Idx && poly2DLList[DLLVertexCountOffsetIdx + inextIdx].linear_distance <
                                                       tempNodeB.linear_distance)
                    {
                        inextIdx = poly2DLList[DLLVertexCountOffsetIdx + inextIdx].nextIdx;
                    }
                    iprevIdx = poly2DLList[DLLVertexCountOffsetIdx + inextIdx].prevIdx;

                    // Insert node1 between iprev and inext
                    poly2DLList[DLLVertexCountOffsetIdx + inextIdx].prevIdx = DLLPoly2ElementCount;
                    tempNodeB.nextIdx = inextIdx;
                    tempNodeB.prevIdx = iprevIdx;
                    poly2DLList[DLLVertexCountOffsetIdx + iprevIdx].nextIdx = DLLPoly2ElementCount;

                    // Insert the nodes into the result array (append them, the pointers are correctly chained)
                    poly1DLList[DLLVertexCountOffsetIdx + DLLPoly1ElementCount] = tempNodeA;
                    poly2DLList[DLLVertexCountOffsetIdx + DLLPoly2ElementCount] = tempNodeB;

                    // Increment the total number of vertices
                    DLLPoly1ElementCount++;
                    DLLPoly2ElementCount++;
                }

                // Find the next non intersection vertex
                do
                {
                    here2Idx = poly2DLList[DLLVertexCountOffsetIdx + here2Idx].nextIdx;
                } while (poly2DLList[DLLVertexCountOffsetIdx + here2Idx].is_intersect != false);

            } while (here2Idx != ROOT_NODE_IDX);

            // Update vertex
            // Find the next non intersection vertex
            do
            {
                here1Idx = poly1DLList[DLLVertexCountOffsetIdx + here1Idx].nextIdx;
            } while (poly1DLList[DLLVertexCountOffsetIdx + here1Idx].is_intersect != false);

        } while (here1Idx != ROOT_NODE_IDX);


		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Calculate if the first vertex of a polygon is in the other polygon or not
		// We calculate for the root element of both polygons

		// Poly 1 root in poly 2
		bool is1in2 = false;
        float x = poly1DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].point.latitude;
        float y = poly1DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].point.longitude;
        int32_t hereIdx = ROOT_NODE_IDX;
        int32_t nextIdx = ROOT_NODE_IDX;

        do
        {
            nextIdx = poly2DLList[DLLVertexCountOffsetIdx + hereIdx].nextIdx;
            float hx = poly2DLList[DLLVertexCountOffsetIdx + hereIdx].point.latitude;
            float hy = poly2DLList[DLLVertexCountOffsetIdx + hereIdx].point.longitude;
            float nx = poly2DLList[DLLVertexCountOffsetIdx + nextIdx].point.latitude;
            float ny = poly2DLList[DLLVertexCountOffsetIdx + nextIdx].point.longitude;
            if (((hy < y && ny >= y) || (hy >= y && ny < y)) && (hx <= x || nx <= x) &&
                (hx + (y - hy) / (ny - hy) * (nx - hx) < x))
            {
                is1in2 = !is1in2;
            }
            hereIdx = nextIdx;
        } while (hereIdx != ROOT_NODE_IDX);

        // Poly 1 root in poly 2
        bool is2in1 = false;
        x = poly2DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].point.latitude;
        y = poly2DLList[DLLVertexCountOffsetIdx + ROOT_NODE_IDX].point.longitude;
        hereIdx = ROOT_NODE_IDX;
        nextIdx = ROOT_NODE_IDX;

        do
        {
            nextIdx = poly1DLList[DLLVertexCountOffsetIdx + hereIdx].nextIdx;
            float hx = poly1DLList[DLLVertexCountOffsetIdx + hereIdx].point.latitude;
            float hy = poly1DLList[DLLVertexCountOffsetIdx + hereIdx].point.longitude;
            float nx = poly1DLList[DLLVertexCountOffsetIdx + nextIdx].point.latitude;
            float ny = poly1DLList[DLLVertexCountOffsetIdx + nextIdx].point.longitude;
            if (((hy < y && ny >= y) || (hy >= y && ny < y)) && (hx <= x || nx <= x) &&
                (hx + (y - hy) / (ny - hy) * (nx - hx) < x))
            {
                is2in1 = !is2in1;
            }
            hereIdx = nextIdx;
        } while (hereIdx != ROOT_NODE_IDX);

		// Now label the is_entry attributein a zig-zag pattern based on the first point calculated above
		// Poly 1
        hereIdx = ROOT_NODE_IDX;
        bool isEntry = !is1in2;
        do
        {
            if (poly1DLList[DLLVertexCountOffsetIdx + hereIdx].is_intersect)
            {
                poly1DLList[DLLVertexCountOffsetIdx + hereIdx].is_entry = isEntry;
                isEntry = !isEntry;
            }
            hereIdx = poly1DLList[DLLVertexCountOffsetIdx + hereIdx].nextIdx;
        } while (hereIdx != ROOT_NODE_IDX);

		// Poly 2
        hereIdx = ROOT_NODE_IDX;
        isEntry = !is2in1;
        do
        {
            if (poly2DLList[DLLVertexCountOffsetIdx + hereIdx].is_intersect)
            {
                poly2DLList[DLLVertexCountOffsetIdx + hereIdx].is_entry = isEntry;
                isEntry = !isEntry;
            }
            hereIdx = poly2DLList[DLLVertexCountOffsetIdx + hereIdx].nextIdx;
        } while (hereIdx != ROOT_NODE_IDX);
        //////////////////////////////////////////////////////////////////////////////////////////////////
		// Now reconstruct the polygons based on edge tracing and the desired functors
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
