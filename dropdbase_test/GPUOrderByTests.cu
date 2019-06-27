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
    const int32_t columnCount = 4;
    const int32_t dataElementCount = 16;

    // Input data
    std::vector<std::vector<int32_t>>unsigned_integer_columns_in = {
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
        {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4},
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}
    };

    // Output data
    std::vector<int32_t> unsigned_integers_out_1(dataElementCount);
    std::vector<int32_t> unsigned_integers_out_2(dataElementCount);
    std::vector<int32_t> unsigned_integers_out_3(dataElementCount);
    std::vector<int32_t> unsigned_integers_out_4(dataElementCount);

    std::vector<OrderBy::Order> order = {
        OrderBy::Order::ASC,
        OrderBy::Order::DESC,
        OrderBy::Order::ASC,
        OrderBy::Order::ASC
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Input buffers
    std::vector<int32_t*> d_unsigned_integer_columns_in(columnCount);

    // Output buffers
    int32_t* d_unsigned_integers_out;

    // Reordered output d_indices
    int32_t* d_indices;

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
    GPUMemory::alloc(&d_indices,  dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Run the order by operation
    GPUOrderBy<int32_t> ob(dataElementCount);

    ob.OrderBy(d_indices, d_unsigned_integer_columns_in, dataElementCount, order);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy back the results and print them

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices, d_unsigned_integer_columns_in[0], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_1[0], d_unsigned_integers_out, dataElementCount);

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices, d_unsigned_integer_columns_in[1], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_2[0], d_unsigned_integers_out, dataElementCount);

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices, d_unsigned_integer_columns_in[2], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_3[0], d_unsigned_integers_out, dataElementCount);

    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices, d_unsigned_integer_columns_in[3], dataElementCount);
    GPUMemory::copyDeviceToHost(&unsigned_integers_out_4[0], d_unsigned_integers_out, dataElementCount);

    for(int32_t i = 0; i < dataElementCount; i++)
    {
        std::printf("%2d %2d %2d %2d\n", 
        unsigned_integers_out_1[i], 
        unsigned_integers_out_2[i], 
        unsigned_integers_out_3[i], 
        unsigned_integers_out_4[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Free the input buffers
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::free(d_unsigned_integer_columns_in[i]);
    }

    GPUMemory::free(d_indices);
    GPUMemory::free(d_unsigned_integers_out);
}
