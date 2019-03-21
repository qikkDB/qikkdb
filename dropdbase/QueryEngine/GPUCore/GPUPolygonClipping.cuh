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
    __device__ __host__ void operator()(bool* into) const
    {
        into[0] = true;
        into[1] = true;
    }
};

struct polyUnion
{
    __device__ __host__ void operator()(bool* into) const
    {
        into[0] = false;
        into[1] = false;
    }
};
} // namespace PolygonFunctions

// Struct for the polygon Doubly Linked List construction on the GPU
__device__ struct PolygonNodeDLL
{
    NativeGeoPoint point;

    float linear_distance;
    bool is_intersect;
    bool is_entry;
    bool is_processed;

    int32_t nextIdx;
    int32_t prevIdx;
    int32_t cross_link; // Cross ink also indicates if a node is a intersect or not = only intersects have positive cross links
};

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
                                        int32_t dataElementCount,
                                        int32_t* poly1VertexCounts,
                                        int32_t* poly2VertexCounts,
                                        int32_t* DLLVertexCounts,
                                        int32_t* DLLVertexCountOffsets,
                                        int32_t* DLLPolygonCounts,
                                        int32_t* DLLPolygonCountOffsets,
                                        PolygonNodeDLL* poly1DLList,
                                        PolygonNodeDLL* poly2DLList)
{
    // The root of the DLL is always the 0th element
    const int32_t ROOT_NODE_IDX = 0;

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Transform the input polygons to Doubly Linked Lists (DLLs)
        // Only for 0th polygon

        // Get the offset index for the dynamic list index - needed because of the character of the prefix sum

        int32_t DLLVertexCountOffsetIdx = 0; // Base offset in the vertex array
        int32_t DLLPolygonCountOffsetIdx = 0; // Base offset in the polygon array

        if (i > 0)
        {
            DLLVertexCountOffsetIdx = DLLVertexCountOffsets[i - 1];
            DLLPolygonCountOffsetIdx = DLLPolygonCountOffsets[i - 1];
        }


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointers to the end of the DLL lists ( one element after the last element)
        int32_t DLLPoly1ElementCount = 0;
        int32_t DLLPoly2ElementCount = 0;

        // Poly 1
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

            tempNode.point = {x, y};

            tempNode.linear_distance = 0;
            tempNode.is_intersect = false;
            tempNode.is_entry = false;
            tempNode.is_processed = false;

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

            tempNode.point = {x, y};

            tempNode.linear_distance = 0;
            tempNode.is_intersect = false;
            tempNode.is_entry = false;
            tempNode.is_processed = false;

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
                float alongA = 0;
                float alongB = 0;

                float point_x = 0;
                float point_y = 0;

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

                    tempNodeA.point = {point_x, point_y};

                    tempNodeA.linear_distance = alongA;
                    tempNodeA.is_intersect = true;
                    tempNodeA.is_entry = false;
                    tempNodeA.is_processed = false;

                    tempNodeA.cross_link = DLLPoly2ElementCount;

                    // Create the intersect node in second list
                    PolygonNodeDLL tempNodeB;

                    tempNodeB.point = {point_x, point_y};

                    tempNodeB.linear_distance = alongB;
                    tempNodeB.is_intersect = true;
                    tempNodeB.is_entry = false;
                    tempNodeB.is_processed = false;

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

        // Now label the is_entry attributein a zig-zag pattern based on the first point calculated
        // above Poly 1
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

        // Counter for vertices in the resulting polygon
        // Counter for polygons in resulting polygon
        int32_t VertexCountOutPoly = 0;
        int32_t PolygonCountOutPoly = 0;

        int32_t VertexOffsetOutPoly = 0;
        int32_t PolygonOffsetOutPoly = 0;


        // Process all intersects based on the chosen functor
        int32_t isectIdx = ROOT_NODE_IDX;
        while (true)
        {
            do
            {
                if (poly1DLList[DLLVertexCountOffsetIdx + isectIdx].is_intersect &&
                    !poly1DLList[DLLVertexCountOffsetIdx + isectIdx].is_processed)
                {
                    // If the vertex is an intersection and it was not processed then process it, else skip
                    break;
                }
                isectIdx = poly1DLList[DLLVertexCountOffsetIdx + isectIdx].nextIdx;
            } while (isectIdx != ROOT_NODE_IDX);

            if (isectIdx == ROOT_NODE_IDX)
            {
                // If we iterated over the whole list - then exit
                // TODO I am not sure about this - test it !
                break;
            }

            // Process isect
            // false false - union
            // true true - interset
            bool into[2]; //{true, true};
            OP{}(into); // Assign operation based on functor

            int32_t curpoly = 0;
            bool moveForward = false;

            bool intersectFound = false;
            bool allProcessed = false;

            int32_t hereClipIdx = isectIdx;

            // Zero the output polygon vertex counter
            VertexCountOutPoly = 0;

            do
            {

                // Switch base on which pointer is active

                if (curpoly == 0)
                {
                    // Mark the found intersection as processed both on the current list and the cross link list
                    poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_processed = true;
                    poly2DLList[DLLVertexCountOffsetIdx +
                                poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].cross_link]
                        .is_processed = true;

                    moveForward =
                        (poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_entry == into[curpoly]);
                }
                else if (curpoly == 1)
                {
                    // Mark the found intersection as processed both on the current list and the cross link list
                    poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_processed = true;
                    poly1DLList[DLLVertexCountOffsetIdx +
                                poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].cross_link]
                        .is_processed = true;

                    moveForward =
                        (poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_entry == into[curpoly]);
                }

                do
                {
                    // Save the point as a new result point for the result polygon
                    float lat;
                    float lon;
                    if (curpoly == 0)
                    {
                        lat = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].point.latitude;
                        lon = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].point.longitude;
                    }
                    else if (curpoly == 1)
                    {
                        lat = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].point.latitude;
                        lon = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].point.longitude;
                    }

                    // Write output data to correct offsets
                    complexPolygonOut
                        .polyPoints[DLLVertexCountOffsetIdx + VertexOffsetOutPoly + VertexCountOutPoly]
                        .latitude = lat;
                    complexPolygonOut
                        .polyPoints[DLLVertexCountOffsetIdx + VertexOffsetOutPoly + VertexCountOutPoly]
                        .longitude = lon;

                    // Increment vertex count
                    VertexCountOutPoly++;

                    // Move according to in/out status
                    if (moveForward)
                    {
                        if (curpoly == 0)
                        {
                            hereClipIdx = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].nextIdx;
                        }
                        else if (curpoly == 1)
                        {
                            hereClipIdx = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].nextIdx;
                        }
                    }
                    else
                    {
                        if (curpoly == 0)
                        {
                            hereClipIdx = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].prevIdx;
                        }
                        else if (curpoly == 1)
                        {
                            hereClipIdx = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].prevIdx;
                        }
                    }

                    // Do this until an intersection is found
                    if (curpoly == 0)
                    {
                        intersectFound = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_intersect;
                    }
                    else if (curpoly == 1)
                    {
                        intersectFound = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_intersect;
                    }

                } while (!intersectFound);

                // We've hit the next intersection so switch polygons
                if (curpoly == 0)
                {
                    hereClipIdx = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].cross_link;
                }
                else if (curpoly == 1)
                {
                    hereClipIdx = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].cross_link;
                }
                curpoly = 1 - curpoly;

                // Do this until all vertices for a result polygon are processed
                if (curpoly == 0)
                {
                    allProcessed = poly1DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_processed;
                }
                else if (curpoly == 1)
                {
                    allProcessed = poly2DLList[DLLVertexCountOffsetIdx + hereClipIdx].is_processed;
                }
            } while (!allProcessed);
            // Save the reults for reconstruction
            complexPolygonOut.pointCount[DLLPolygonCountOffsetIdx + PolygonCountOutPoly] = VertexCountOutPoly;

            // Icrement the polygon offset
            VertexOffsetOutPoly += VertexCountOutPoly;
            PolygonCountOutPoly++;
        }
        complexPolygonOut.polyCount[i] = PolygonCountOutPoly;
    }
}

// Offset the inclusive sum to a exclusive sum
template <typename T>
__global__ void kernel_transform_inclusive_to_exclusive_sum(T* in, T* out, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (i <= 0)
            in[i] = 0;
        else
            in[i] = out[i - 1];
    }
}

// Compress the poygin/vertex counts into one single array based on the initial sandbox offsets
// This function is set up to accept inclusive prefix sums only !!!
template <typename T>
__global__ void kernel_compress_based_on_offset_element_counts_inclusive(T* outCompressedData,
                                                                         T* inUncompressedData,
                                                                         int32_t* inCompressedOffsets,
                                                                         int32_t* inUncompressedOffsets,
                                                                         int32_t* inCounts,
                                                                         int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int32_t inUncompressedOffset = 0;
        int32_t inCompressedOffset = 0;

        // Preform inclusive to exclusive index switch
        if (i > 0)
        {
            inUncompressedOffset = inUncompressedOffsets[i - 1];
            inCompressedOffset = inCompressedOffsets[i - 1];
        }

        // Compress the data
        for (int j = 0; j < inCounts[i]; j++)
        {
            outCompressedData[inCompressedOffset + j] = inUncompressedData[inUncompressedOffset + j];
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

        // Vertex counts in each input polygon - needed for offset calculation
        int32_t* poly1VertexCounts;
        int32_t* poly2VertexCounts;

        // Vertex count start offsets
        // int32_t* poly1VertexOffsets;
        // int32_t* poly2VertexOffsets;

        // Doubly linked list offset buffers
        int32_t* DLLVertexCounts;
        int32_t* DLLVertexCountOffsets;

        int32_t* DLLPolygonCounts;
        int32_t* DLLPolygonCountOffsets;

        // Data buffers for doubly linked lists of polygons during clipping
        PolygonNodeDLL* poly1DLList;
        PolygonNodeDLL* poly2DLList;

        // Total/Maximal size of the result DLL list vertices and polygons
        int32_t DLLVertexCountTotal;
        int32_t DLLPolygonCountTotal;

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

        DLLPolygonCounts = nullptr;
        GPUMemory::allocAndSet(&DLLPolygonCounts, 0, dataElementCount);

        // Calc (n*k + n + k)/3 for each input row
        GPUArithmetic::colConst<ArithmeticOperations::div>(DLLPolygonCounts, DLLVertexCounts, 3, dataElementCount);

        // Calc the prefix sum for each input row - polygons
        DLLPolygonCountOffsets = nullptr;
        GPUMemory::allocAndSet(&DLLPolygonCountOffsets, 0, dataElementCount);

        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLPolygonCounts,
                                      DLLPolygonCountOffsets, dataElementCount);
        GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, DLLPolygonCounts,
                                      DLLPolygonCountOffsets, dataElementCount);
        GPUMemory::free(d_temp_storage);

        // Get the number of maximum polygon count for the result buffer allocation
        DLLPolygonCountTotal = 0;

        GPUMemory::copyDeviceToHost(&DLLPolygonCountTotal, DLLPolygonCountOffsets + dataElementCount - 1, 1);

        //////
        // DEBUG
        // int32_t offsets3[1];
        // GPUMemory::copyDeviceToHost(offsets3, DLLPolygonCountOffsets, 1);

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
        GPUMemory::alloc(&polygonOutTemp.pointIdx, DLLPolygonCountTotal);
        GPUMemory::alloc(&polygonOutTemp.pointCount, DLLPolygonCountTotal);

        // Run the clipping kernel
        kernel_polygon_clipping<OP><<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            polygonOutTemp, polygon1, polygon2, dataElementCount, poly1VertexCounts,
            poly2VertexCounts, DLLVertexCounts, DLLVertexCountOffsets, DLLPolygonCounts,
            DLLPolygonCountOffsets, poly1DLList, poly2DLList);

        ///////////////////////////////////////////////////////////////////////////////////////
        // Reconstruct the real output polygon from temp output polygon
        // Use the tempOut poly as a temporal buffer
        // First the complex polygon layer is reconstructed

        // Alloc the complexPolygon count buffer and the complex polygon idx buffer
        GPUMemory::alloc(&polygonOut.polyCount, dataElementCount);
        GPUMemory::alloc(&polygonOut.polyIdx, dataElementCount);

        // Copy the count buffers from the temp polygon and calculate an exclusive prefix sum to the idx buffer
        GPUMemory::copyDeviceToDevice(polygonOut.polyCount, polygonOutTemp.polyCount, dataElementCount);

        // Calculate the exclusive scan for the complex polygon id buffe
        // First calculate the inclusive scan to get the ount of the complex polygons
        // Then transform it to an exclusive scan to the final output
        // The complex polygon count is used for the next step in poly reconstruction
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, polygonOutTemp.polyCount,
                                      polygonOutTemp.polyIdx, dataElementCount);
        GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, polygonOutTemp.polyCount,
                                      polygonOutTemp.polyIdx, dataElementCount);
        GPUMemory::free(d_temp_storage);

        // Get the count of the output polygons
        int32_t complexPolygonOutCount = 0;
        GPUMemory::copyDeviceToHost(&complexPolygonOutCount, polygonOutTemp.polyIdx + dataElementCount - 1, 1);

        // Transfer the idx prefix sum to the final output as an exclusive scan
        kernel_transform_inclusive_to_exclusive_sum<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            polygonOut.polyIdx, polygonOutTemp.polyIdx, dataElementCount);

        // Alloc the point count buffer and the point idx buffer based on the previously calculated sum
        GPUMemory::alloc(&polygonOut.pointCount, complexPolygonOutCount);
        GPUMemory::alloc(&polygonOut.pointIdx, complexPolygonOutCount);

		// A helper buffer for inclusive to exclusive prefix sum transfer
        int32_t *tempPointIdxBuffer;
        GPUMemory::alloc(&tempPointIdxBuffer, complexPolygonOutCount);

        // Compress the polygons/ point counts
        kernel_compress_based_on_offset_element_counts_inclusive<<<context.calcGridDim(dataElementCount),
                                                         context.getBlockDim()>>>(
            polygonOut.pointCount, polygonOutTemp.pointCount, polygonOutTemp.polyIdx,
            DLLPolygonCountOffsets, polygonOutTemp.polyCount, dataElementCount);

		// Calculate the  inclusive prefix sum for the points ( same as above), retrieve the size then transfer to exclusive prefix sum
		d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, polygonOut.pointCount,
                                      tempPointIdxBuffer, dataElementCount);
        GPUMemory::alloc(reinterpret_cast<int8_t**>(&d_temp_storage), temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, polygonOutTemp.pointCount,
                                      tempPointIdxBuffer, dataElementCount);
        GPUMemory::free(d_temp_storage);

        // Get the count of the output polygons
        int32_t pointOutCount = 0;
        GPUMemory::copyDeviceToHost(&pointOutCount, tempPointIdxBuffer + dataElementCount - 1, 1);

		// Transform the inclusive sum of points to exclusive sum
        kernel_transform_inclusive_to_exclusive_sum<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            polygonOut.pointIdx, tempPointIdxBuffer, complexPolygonOutCount);

		// Alloc the array of output pointsbased on the retrieved size
        GPUMemory::alloc(&polygonOut.polyPoints, pointOutCount);

        // Compress the array of output points
        kernel_compress_based_on_offset_element_counts_inclusive<<<context.calcGridDim(dataElementCount),
                                                         context.getBlockDim()>>>(
            polygonOut.polyPoints, polygonOutTemp.polyPoints, tempPointIdxBuffer,
            DLLVertexCountOffsets, polygonOutTemp.pointCount, complexPolygonOutCount);

		/*
        // DEBUG START //
        // Temp
        NativeGeoPoint* res = new NativeGeoPoint[pointOutCount];
        int32_t *complexPolygonIdxRes = new int32_t[dataElementCount];
        int32_t* complexPolygonCntRes = new int32_t[dataElementCount];
        int32_t* polygonIdxRes = new int32_t[complexPolygonOutCount];
        int32_t* polygonCntRes = new int32_t[complexPolygonOutCount];

        GPUMemory::copyDeviceToHost(res, polygonOut.polyPoints, pointOutCount);
        GPUMemory::copyDeviceToHost(complexPolygonIdxRes, polygonOut.polyIdx, dataElementCount);
        GPUMemory::copyDeviceToHost(complexPolygonCntRes, polygonOut.polyCount, dataElementCount);
        GPUMemory::copyDeviceToHost(polygonIdxRes, polygonOut.pointIdx, complexPolygonOutCount);
        GPUMemory::copyDeviceToHost(polygonCntRes, polygonOut.pointCount, complexPolygonOutCount);
        
        printf("\n\nVertices\n");
        for (int s = 0; s < pointOutCount; s++)
        {
            printf("[%.2f,%.2f],\n", res[s].latitude, res[s].longitude);
        }

        printf("\n\nPoly counts\n");
        for (int s = 0; s < dataElementCount; s++)
        {
            printf("%d\n", complexPolygonCntRes[s]);
        }

		 printf("\n\nPoly idx\n");
        for (int s = 0; s < dataElementCount; s++)
        {
            printf("%d\n", complexPolygonIdxRes[s]);
        }

        printf("\n\nPoint counts\n");
        for (int s = 0; s < complexPolygonOutCount; s++)
        {
            printf("%d,\n", polygonCntRes[s]);
        }

		printf("\n\nPoint idx\n");
        for (int s = 0; s < complexPolygonOutCount; s++)
        {
            printf("%d,\n", polygonIdxRes[s]);
        }

		delete[] res;
        delete[] complexPolygonIdxRes;
        delete[] complexPolygonCntRes;
        delete[] polygonIdxRes;
        delete[] polygonCntRes;
        // DEBUG END //
		*/

        // Free the tempora polygon
        GPUMemory::free(polygonOutTemp.polyPoints);
        GPUMemory::free(polygonOutTemp.polyIdx);
        GPUMemory::free(polygonOutTemp.polyCount);
        GPUMemory::free(polygonOutTemp.pointIdx);
        GPUMemory::free(polygonOutTemp.pointCount);

        // Free the allocated memory
        GPUMemory::free(poly1VertexCounts);
        GPUMemory::free(poly2VertexCounts);

        GPUMemory::free(poly1DLList);
        GPUMemory::free(poly2DLList);

        GPUMemory::free(DLLVertexCounts);
        GPUMemory::free(DLLVertexCountOffsets);

		GPUMemory::free(tempPointIdxBuffer);

        // Set error
        QueryEngineError::setCudaError(cudaGetLastError());
    }
};
