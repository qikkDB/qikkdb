
#include "gtest/gtest.h"
#include "../qikkDB/QueryEngine/Context.h"
#include "../qikkDB/QueryEngine/GPUCore/GPUMemory.cuh"
#include "../qikkDB/QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../qikkDB/QueryEngine/OrderByType.h"

#include <vector>
#include <cstdint>
#include <iostream>
#include <functional>
#include <chrono>

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

template<typename T> 
void OrderByTestTemplate(int32_t colCount, 
                         int32_t dataElementCount, 
                         int32_t numericDataLimit,
                         bool performCPUTest = true,
                         bool performGPUTest = true,
                         std::chrono::duration<double> *elapsed = nullptr,
                         bool suppress = true)
{
    // Random generator
    int32_t SEED = 42;
    srand(SEED);

    bool PERFORM_CPU_TEST = performCPUTest;
    bool PERFORM_GPU_TEST = performGPUTest;

    // Input sizes
    int32_t COL_COUNT = colCount;
    int32_t COL_DATA_ELEMENT_COUNT = dataElementCount;

    uint32_t NUMERIC_DATA_LIMIT = numericDataLimit;

    // Input data
    std::vector<OrderBy::Order> orderingIn;
    std::vector<std::vector<T>> dataIn;
    std::vector<std::vector<T>> dataOut;
    std::vector<std::vector<T>> dataOutGPU;

    // Fill the input data vectors
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        orderingIn.push_back((rand() % 2) == 0 ? OrderBy::Order::ASC : OrderBy::Order::DESC);
        dataIn.push_back(std::vector<T>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            dataIn[i].push_back(rand() % NUMERIC_DATA_LIMIT);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////
    if(PERFORM_CPU_TEST)
    {
    // Sort the input data on the CPU
    // This is Done by a new algorithm where we want to sort the input columns accross
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
    std::vector<T> data(COL_DATA_ELEMENT_COUNT);

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
        std::vector<IdxKeyPair<T>> v(COL_DATA_ELEMENT_COUNT);
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            v[j] = {indices[j], data[j]};
        }

        if(orderingIn[i] == OrderBy::Order::ASC)
        {
            stable_sort(v.begin(), v.end(), Asc<T>());
        }
        else 
        {
            stable_sort(v.begin(), v.end(), Desc<T>());
        }

        // 5. Keep the new index combination
        for (int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            indices[j] = v[j].index; 
        }
    }

    // 6. Write the results
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        dataOut.push_back(std::vector<T>{});
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            // 7. Reaorer by the final indices list
            dataOut[i].push_back(dataIn[i][indices[j]]);
        }
    }
    
    // DEBUG
    // std::printf("###############################################################\n");
    // std::printf("### CPU ORDER BY ###\n");
    // // Print the results as columns
    // for(int32_t i = 0; i < COL_COUNT; i++)
    // {
    //     std::printf("%2c ", orderingIn[i] == OrderBy::Order::ASC ? 'A' : 'D');
    // }
    // std::printf("\n");

    // for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    // {
    //     for(int32_t j = 0; j < COL_COUNT; j++)
    //     {
    //         std::printf("%2u ", dataOut[j][i]);
    //     }
    //     std::printf("\n");
    // }
    //DEBUG END


    }
    /////////////////////////////////////////////////////////////////////////////
    if(PERFORM_GPU_TEST)
    {
    // Sort the input data on the GPU
    std::vector<T*> d_dataIn;
    int32_t* d_indexBuffer;
    T* d_resultBuffer;

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
    GPUOrderBy ob(COL_DATA_ELEMENT_COUNT);

    for(int32_t i = d_dataIn.size() - 1; i >= 0; i--)
    {
        ob.OrderByColumn(d_indexBuffer, d_dataIn[i], nullptr, COL_DATA_ELEMENT_COUNT, orderingIn[i]);
    }
    
    // Copy back the results
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        // Reconstruct the data
        ob.ReOrderByIdx(d_resultBuffer, d_indexBuffer, d_dataIn[i], COL_DATA_ELEMENT_COUNT);

        // Copy back the data
        dataOutGPU.push_back(std::vector<T>(COL_DATA_ELEMENT_COUNT));
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
    
    // // DEBUG
    // std::printf("###############################################################\n");
    // std::printf("### GPU ORDER BY ###\n");
    // // Print the results as columns
    // for(int32_t i = 0; i < COL_COUNT; i++)
    // {
    //     std::printf("%2c ", orderingIn[i] == OrderBy::Order::ASC ? 'A' : 'D');
    // }
    // std::printf("\n");

    // for(int32_t i = 0; i < COL_DATA_ELEMENT_COUNT; i++)
    // {
    //     for(int32_t j = 0; j < COL_COUNT; j++)
    //     {
    //         std::printf("%2u ", dataOutGPU[j][i]);
    //     }
    //     std::printf("\n");
    // }
    // //DEBUG END
    
    }
    /////////////////////////////////////////////////////////////////////////////
    auto finish = std::chrono::high_resolution_clock::now();

    if(PERFORM_CPU_TEST && PERFORM_GPU_TEST)
    {
    // Compare the data
    for(int32_t i = 0; i < COL_COUNT; i++)
    {
        for(int32_t j = 0; j < COL_DATA_ELEMENT_COUNT; j++)
        {
            if (std::is_integral<T>::value)
            {
                ASSERT_EQ(dataOut[i][j], dataOutGPU[i][j]);
            }
            else
            {
                ASSERT_FLOAT_EQ(dataOut[i][j], dataOutGPU[i][j]);
            }
        }
    }
    }
    else
    {
        if(!suppress)
        {
            std::cout << "[WARNING] Skipped GPU or CPU computation , result comparsions halted " << std::endl;
        }
    }

    if(elapsed != nullptr)
    {
        *elapsed = finish - start;
    }
    
}

class GPUOrderByTests : public ::testing::Test
{
public:
    // Input sizes
    int32_t COL_COUNT;
    int32_t COL_DATA_ELEMENT_COUNT;

    uint32_t NUMERIC_DATA_LIMIT;

    virtual void SetUp()
    {
        COL_COUNT = 5;
        COL_DATA_ELEMENT_COUNT = 1 << 10;
    
        NUMERIC_DATA_LIMIT = 10000;
    }

	virtual void TearDown()
	{
	}
};

TEST_F(GPUOrderByTests, GPUOrderByUnsigned32Test)
{
    OrderByTestTemplate<uint32_t>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderBySigned32Test)
{
    OrderByTestTemplate<int32_t>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderByUnsigned64Test)
{
    OrderByTestTemplate<uint64_t>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderBySigned64Test)
{
    OrderByTestTemplate<int64_t>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderByFloatTest)
{
    OrderByTestTemplate<float>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderByDoubleTest)
{
    OrderByTestTemplate<double>(COL_COUNT, COL_DATA_ELEMENT_COUNT, NUMERIC_DATA_LIMIT);
}

TEST_F(GPUOrderByTests, GPUOrderByTimingTest)
{
    /*
    int32_t s_colCount = 1;
    int32_t e_colCount = 10;
    int32_t i_colCount = 1;

    int32_t s_dataElementCountExp = 10;
    int32_t e_dataElementCountExp = 27;
    int32_t i_dataElementCountExp = 1;

    int32_t NUMERIC_DATA_LIMIT = 10000;
    // int32_t s_numericDataLimit;
    // int32_t e_numericDataLimit;
    // int32_t i_numericDataLimit;
    
    //Try the CPU calcualtion for different column sizes and column count
    std::printf("CPU  ");
    for(int32_t colCount = s_colCount; colCount <= e_colCount; colCount += i_colCount)
    {
        std::printf("%10d ", colCount);
    }
    std::printf("\n");

    for(int32_t dataElementCountExp = s_dataElementCountExp; dataElementCountExp <= e_dataElementCountExp; dataElementCountExp += i_dataElementCountExp)
    {
        std::printf("2^%2d ", dataElementCountExp);
        for(int32_t colCount = s_colCount; colCount <= e_colCount; colCount += i_colCount)
        {
            std::chrono::duration<double> elapsed;
            OrderByTestTemplate<int32_t>(colCount, 1 << dataElementCountExp, NUMERIC_DATA_LIMIT, true, false, &elapsed);
            std::printf("%10d ", (int32_t)(1000 * elapsed.count()));
        }
        std::printf("\n");
    }
    
    std::printf("\n");
    std::printf("\n");

    

    //Try the GPU calcualtion for different column sizes and column count
    std::printf("GPU  ");
    for(int32_t colCount = s_colCount; colCount <= e_colCount; colCount += i_colCount)
    {
        std::printf("%10d ", colCount);
    }
    std::printf("\n");

    for(int32_t dataElementCountExp = s_dataElementCountExp; dataElementCountExp <= e_dataElementCountExp; dataElementCountExp += i_dataElementCountExp)
    {
        std::printf("2^%2d ", dataElementCountExp);
        for(int32_t colCount = s_colCount; colCount <= e_colCount; colCount += i_colCount)
        {
            std::chrono::duration<double> elapsed;
            OrderByTestTemplate<int32_t>(colCount, 1 << dataElementCountExp, NUMERIC_DATA_LIMIT, false, true, &elapsed);
            std::printf("%10d ", (int32_t)(1000 * elapsed.count()));
        }
        std::printf("\n");
    }
    */
}