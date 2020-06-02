#include "../qikkDB/ColumnBase.h"
#include "../qikkDB/ComplexPolygonFactory.h"
#include "../qikkDB/Database.h"
#include "../qikkDB/DatabaseGenerator.h"
#include "../qikkDB/PointFactory.h"
#include "../qikkDB/QueryEngine/GPUMemoryCache.h"
#include "gtest/gtest.h"

TEST(AllocatorTests, AllocateDeallocate)
{
    const int SIZE = 2048;

    auto& context = Context::getInstance();

    auto& allocator = context.GetAllocatorForCurrentDevice();

    std::vector<int8_t*> pointers;

    pointers.push_back(allocator.Allocate(SIZE));
    pointers.push_back(allocator.Allocate(SIZE));

    // check if it is correct:
    for (int i = 0; i < pointers.size(); i++)
    {
        for (int j = 0; j < pointers.size(); j++)
        {
            if (i != j)
            {
                ASSERT_TRUE(pointers[j] >= pointers[i] + SIZE ||
                            (pointers[j] < pointers[i] && pointers[i] >= pointers[j] + SIZE));
            }
        }
    }

    pointers.push_back(allocator.Allocate(SIZE));
    allocator.Deallocate(pointers[0]);

    // check if it is correct:
    for (int i = 0; i < pointers.size(); i++)
    {
        for (int j = 0; j < pointers.size(); j++)
        {
            if (i != j)
            {
                ASSERT_TRUE(pointers[j] >= pointers[i] + SIZE ||
                            (pointers[j] < pointers[i] && pointers[i] >= pointers[j] + SIZE));
            }
        }
    }
}