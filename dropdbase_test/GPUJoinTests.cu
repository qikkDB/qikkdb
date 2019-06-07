#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"

TEST(GPUJoinTests, JoinTest)
{
    // Initialize CUDA context:
    Context::getInstance();

}