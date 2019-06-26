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
    const int32_t columnCount = 1;
    const int32_t dataElementCount = 16;

    // Input data
    std::vector<std::vector<uint32_t>>unsigned_integer_columns_in = {
        {23, 11, 67, 3, 87, 34, 764, 76, 3, 58, 92, 42, 19, 37, 76, 85}
    };

    // Output data
    std::vector<uint32_t> unsigned_integers_out(dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Input buffers
    std::vector<uint32_t*> d_unsigned_integer_columns_in(columnCount);

    // Output buffers
    uint32_t* d_unsigned_integers_out;

    // Reordered output d_indices
    int32_t* d_indices;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Alloc the input buffers
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::alloc(&d_unsigned_integer_columns_in[i],  dataElementCount);
        GPUMemory::copyHostToDevice(d_unsigned_integer_columns_in[i], &unsigned_integer_columns_in[i][0], dataElementCount);
    }

    // Alloc the output buffers
    GPUMemory::alloc(&d_unsigned_integers_out,  dataElementCount);

    // Alloc the d_indices buffer
    GPUMemory::alloc(&d_indices,  dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Run the order by operation
    GPUOrderBy<uint32_t> ob(dataElementCount);

    ob.OrderBy(d_indices, d_unsigned_integer_columns_in, dataElementCount);
    
    ob.ReOrderByIdx(d_unsigned_integers_out, d_indices, d_unsigned_integer_columns_in[0], dataElementCount);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy back the results and print them
    GPUMemory::copyDeviceToHost(&unsigned_integers_out[0], d_unsigned_integers_out, dataElementCount);

    for(int32_t i = 0; i < dataElementCount; i++)
    {
        std::printf("%d ", unsigned_integers_out[i]);
    }
    std::printf("\n");

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Free the input buffers
    for(int32_t i = 0; i < columnCount; i++)
    {
        GPUMemory::free(d_unsigned_integer_columns_in[i]);
    }

    GPUMemory::free(d_unsigned_integers_out);
    GPUMemory::free(d_indices);
}
