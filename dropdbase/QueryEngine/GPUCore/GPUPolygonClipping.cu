#include "GPUPolygonClipping.cuh"

/// A kernel for counting the number of vertices that a complex polygon has
__global__ void kernel_calculate_point_count_in_complex_polygon(int32_t* pointCounts,
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
