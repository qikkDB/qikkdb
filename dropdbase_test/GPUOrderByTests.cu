#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUOrderBy.cuh"

#include <vector>
#include <cstdint>
#include <iostream>

TEST(GPUOrderByTests, GPUOrderByTest)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Input data sizes
    const int32_t columnCount = 2;
    const int32_t dataElementCount = 16;

    // Input data
    std::vector<std::vector<uint32_t>>unsigned_integer_columns_in = {
        {56, 65, 34, 87, 99, 87, 56, 99, 56, 87, 59, 36, 65, 99, 56, 34},
        {12, 14, 98, 31, 23, 47, 99, 32, 52, 74, 67, 13, 72, 60, 33, 89}
    };

    // Output data
    std::vector<uint32_t> unsigned_integers_out_1(dataElementCount);
    std::vector<uint32_t> unsigned_integers_out_2(dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Input buffers
    std::vector<uint32_t*> d_unsigned_integer_columns_in(columnCount);

    // Output buffers
    uint32_t* d_unsigned_integers_out;

    // Reordered output d_indices
    std::vector<int32_t*> d_indices(columnCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Alloc the input buffers
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::alloc(&d_unsigned_integer_columns_in[i], dataElementCount);
        GPUMemory::copyHostToDevice(d_unsigned_integer_columns_in[i], &unsigned_integer_columns_in[i][0], dataElementCount);
    }

    // Alloc the output buffers
    GPUMemory::alloc(&d_unsigned_integers_out,  dataElementCount);

    // Alloc the d_indices buffer
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::alloc(&d_indices[i],  dataElementCount);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Run the order by operation
    GPUOrderBy<uint32_t> ob(dataElementCount);

    ob.OrderByAsc(d_indices, d_unsigned_integer_columns_in, dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy back the results and print them

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices[0], d_unsigned_integer_columns_in[0], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_1[0], d_unsigned_integers_out, dataElementCount);

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices[1], d_unsigned_integer_columns_in[1], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_2[0], d_unsigned_integers_out, dataElementCount);

    for(int32_t i = 0; i < dataElementCount; i++)
    {
        std::printf("%2d %2d\n", unsigned_integers_out_1[i], unsigned_integers_out_2[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Free the input buffers
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::free(d_unsigned_integer_columns_in[i]);
        GPUMemory::free(d_indices[i]);
    }

    GPUMemory::free(d_unsigned_integers_out);
}
