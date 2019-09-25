#pragma once

#include "../Context.h"
#include "cuda_ptr.h"
#include "GPUMemory.cuh"

#include "../../ColumnBase.h"
#include "../../BlockBase.h"

#include "../../../cub/cub.cuh"

class MergeJoin
{
    template<typename T> 
	static void Join(ColumnBase<T>& colA, ColumnBase<T>& colB)
    {
    
	}
};