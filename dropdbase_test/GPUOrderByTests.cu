#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUOrderBy.cuh"

#include <vector>
#include <cstdint>
#include <iostream>
#include <functional>


template<typename T>
struct IdxKeyPair
{
    int32_t index;
    T key;
};

template<typename T>
struct Asc
{
    inline bool operator() (const IdxKeyPair<T>& struct1, const IdxKeyPair<T>& struct2)
    {
        return (struct1.key < struct2.key);
    }
};

template<typename T>
struct Desc
{
    inline bool operator() (const IdxKeyPair<T>& struct1, const IdxKeyPair<T>& struct2)
    {
        return (struct1.key > struct2.key);
    }
};

TEST(GPUOrderByTests, GPUOrderByUnsignedTest)
{
    // Random generator
    int32_t SEED = 42;
    srand(SEED);

    bool SKIP_CPU = true;

    // Input sizes
    int32_t COL_COUNT = 1;
    int32_t COL_DATA_ELEMENT_COUNT = 2 << 27;

    uint32_t NUMERIC_DATA_LIMIT = 10;

    // Input data
    std::vector<OrderBy::Order> orderingIn;
    std::vector<std::vector<uint32_t>> dataIn;
    std::vector<std::vector<uint32_t>> dataOut;

    // Fill the input data vectors
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        orderingIn.push_back((rand() % 2) == 0 ? OrderBy::Order::ASC : OrderBy::Order::DESC);
        dataIn.push_back(std::vector<uint32_t>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            dataIn[i].push_back(rand() % NUMERIC_DATA_LIMIT);
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    if(!SKIP_CPU)
    {
    // Sort the input data on the CPU
    // This is done by a new algorithm where we want to sort the input columns accross
    // multiple columns - ORDER BY
    // The algorithm:
    //      Input: A list of columns to be ordered by in that order
    //      Output: A list of indices for the reordered column elements
    //
    //      1. Initialize a list of indices from 0 .. n where n is the number of entries in the input columns
    //      2. Iterate over the columns from the last to the first (important to keep this order), for each column
    //      3. Reorder the column entries into a new index, data pair, each index data pair stores an index and the
    //         corresponding data entry (where the index points) from the input column
    //      4. Sort the index, data pairs based on the data
    //      5. Keep the new index combination for the next iteration
    //      6. If all collumns are processed exit, else go to 3
    //      7. The final indices list is the list of lexicographicsal ordering of all vectors
    //         reorder the input collumns based on these indices to get the order by operation over all columns


    // 0. Create the temporary sort buffers
    std::vector<int32_t> indices(COL_DATA_ELEMENT_COUNT);
    std::vector<uint32_t> data(COL_DATA_ELEMENT_COUNT);

    // 1. Fill in the indices with the default value
    for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    {
        indices[i] = i;
    }

    // 2. Perform the column sorting from the last to the first column
    for(int32_t i = COL_COUNT - 1; i >= 0; i--)
    {
        // 3. Reorder the column entries based on the indices
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            data[j] = dataIn[i][indices[j]];
        }

        // 4. Sort the index-data pairs based on data - mind the ordering
        std::vector<IdxKeyPair<uint32_t>> v(COL_DATA_ELEMENT_COUNT);
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            v[j] = {indices[j], data[j]};
        }

        if(orderingIn[i] == OrderBy::Order::ASC)
        {
            stable_sort(v.begin(), v.end(), Asc<uint32_t>());
        }
        else 
        {
            stable_sort(v.begin(), v.end(), Desc<uint32_t>());
        }

        // 5. Keep the new index combination
        for (int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            indices[j] = v[j].index; 
        }
        std::cout << std:: endl;
    }

    // 6. Write the results
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        dataOut.push_back(std::vector<uint32_t>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            // 7. Reaorer by the final indices list
            dataOut[i].push_back(dataIn[i][indices[j]]);
        }
    }
    
    /*
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
    */

    }
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
    /*
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
    */

    /////////////////////////////////////////////////////////////////////////////
    if(!SKIP_CPU)
    {
    // Compare the data
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            ASSERT_EQ(dataOut[i][j], dataOutGPU[i][j]);
        }
    }
    }
}
