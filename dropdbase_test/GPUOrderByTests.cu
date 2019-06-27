#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUOrderBy.cuh"

#include <vector>
#include <cstdint>
#include <iostream>
#include <functional>

TEST(GPUOrderByTests, GPUOrderByUnsignedTest)
{
    // Random generator
    int32_t SEED = 42;
    srand(SEED);

    // Input sizes
    int32_t COL_COUNT = 6;
    int32_t COL_DATA_ELEMENT_COUNT = 16;

    uint32_t DATA_LIMIT = 10;

    // Input data
    std::vector<OrderBy::Order> orderingIn;
    std::vector<std::vector<uint32_t>> dataIn;

    // Fill the input data vectors
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        orderingIn.push_back(OrderBy::Order::ASC);
        dataIn.push_back(std::vector<uint32_t>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            dataIn[i].push_back(rand() % DATA_LIMIT);
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // Sort the input data on the CPU
    // Fill in the index buffer
    std::vector<int32_t> indexBuffer(COL_DATA_ELEMENT_COUNT);
    for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    {
        indexBuffer[i] = i;
    }

    // Perform the column sorting
    for(int32_t i = COL_COUNT - 1; i >= 0; i--)
    {
        // Fill a vector of index - value pair and sort it based on the indices
        std::vector<std::pair<int32_t, uint32_t>> IKpairs; 
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            IKpairs.push_back(std::make_pair(dataIn[i][indexBuffer[j]], indexBuffer[j]));
        }

        // Sort based on the sort order
        switch(orderingIn[i]){
            case OrderBy::Order::ASC:
                std::sort(IKpairs.begin(), IKpairs.end());
                break;
            case OrderBy::Order::DESC:
                std::sort(IKpairs.begin(), IKpairs.end(), std::greater<>());
                break;
        }

        // Reorder the output vector indices
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            indexBuffer[j] = IKpairs[j].second;
        }
    }

    // Write the results
    std::vector<std::vector<uint32_t>> dataOut;
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        dataOut.push_back(std::vector<uint32_t>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            dataOut[i].push_back(dataIn[i][indexBuffer[j]]);
        }
    }
    
    // DEBUG
    std::printf("###############################################################\n");
    std::printf("### CPU ORDER BY ###\n");
    // Print the results as columns
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        std::printf("%2c ", orderingIn[i] == OrderBy::Order::ASC ? 'A' : 'D');
    }
    std::printf("\n");

    for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    {
        for(int32_t j = 0; j < COL_COUNT; j++)
        {
            std::printf("%2u ", dataOut[j][i]);
        }
        std::printf("\n");
    }
    //DEBUG END

    /////////////////////////////////////////////////////////////////////////////
    // Sort the input data on the GPU
    std::vector<uint32_t*> d_dataIn;
    int32_t* d_indexBuffer;
    uint32_t* d_resultBuffer;

    // Alloc the GPU buffers
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        d_dataIn.push_back(nullptr);
        GPUMemory::alloc(&d_dataIn[i], COL_DATA_ELEMENT_COUNT);
        GPUMemory::copyHostToDevice(d_dataIn[i], &dataIn[i][0], COL_DATA_ELEMENT_COUNT);
    }

    GPUMemory::alloc(&d_indexBuffer, COL_DATA_ELEMENT_COUNT);
    GPUMemory::alloc(&d_resultBuffer, COL_DATA_ELEMENT_COUNT);

    // Perform the orderby operation
    GPUOrderBy<uint32_t> ob(COL_DATA_ELEMENT_COUNT);

    ob.OrderBy(d_indexBuffer, d_dataIn, COL_DATA_ELEMENT_COUNT, orderingIn);
    
    // Copy back the results
    std::vector<std::vector<uint32_t>> dataOutGPU;
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        // Reconstruct the data
        ob.ReOrderByIdx(d_resultBuffer, d_indexBuffer, d_dataIn[i], COL_DATA_ELEMENT_COUNT);

        // Copy back the data
        dataOutGPU.push_back(std::vector<uint32_t>(COL_DATA_ELEMENT_COUNT));
        GPUMemory::copyDeviceToHost(&dataOutGPU[i][0], d_resultBuffer, COL_DATA_ELEMENT_COUNT);
    }

    // Free the GPU buffers
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        GPUMemory::free(d_dataIn[i]);
    }
    GPUMemory::free(d_indexBuffer);
    GPUMemory::free(d_resultBuffer);

    // Print the results
    // DEBUG
    std::printf("###############################################################\n");
    std::printf("### GPU ORDER BY ###\n");
    // Print the results as columns
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        std::printf("%2c ", orderingIn[i] == OrderBy::Order::ASC ? 'A' : 'D');
    }
    std::printf("\n");

    for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    {
        for(int32_t j = 0; j < COL_COUNT; j++)
        {
            std::printf("%2u ", dataOutGPU[j][i]);
        }
        std::printf("\n");
    }
    //DEBUG END

    /////////////////////////////////////////////////////////////////////////////
    // Compare the data

}
