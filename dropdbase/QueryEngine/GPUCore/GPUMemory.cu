#include "GPUMemory.cuh"
#include "../../Types/Point.pb.h"
#include "../../Types/ComplexPolygon.pb.h"

template<>
void GPUMemory::alloc(ColmnarDB::Types::Point **p_Block, int32_t dataElementCount)
{

}

template<>
void GPUMemory::alloc(ColmnarDB::Types::ComplexPolygon **p_Block, int32_t dataElementCount)
{

}