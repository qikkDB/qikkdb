#pragma once
#include <tuple>
#include "Types/ComplexPolygon.pb.h"
#include "NativeGeoPoint.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"


class ComplexPolygonFactory
{
private:
	ComplexPolygonFactory() {};
public:
	static const int MAX_POLYGONS_NUMBER = 8;
	static GPUMemory::GPUPolygon PrepareGPUPolygon(const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons);
	static GPUMemory::GPUPolygon PrepareGPUPolygon(const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, const std::string& columnName, size_t blockIndex);
	static ColmnarDB::Types::ComplexPolygon FromWkt(std::string wkt);
	static std::string WktFromPolygon(const ColmnarDB::Types::ComplexPolygon& polygon);
};