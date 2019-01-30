#pragma once
#include <tuple>
#include "Types/ComplexPolygon.pb.h"
#include "NativeGeoPoint.h"
class ComplexPolygonFactory
{
private:
	ComplexPolygonFactory() {};
public:
	static const int MAX_POLYGONS_NUMBER = 8;
	static std::tuple<std::vector<NativeGeoPoint>, std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>> PrepareGPUPolygon(const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons);
	static ColmnarDB::Types::ComplexPolygon FromWkt(std::string wkt);
	static std::string WktFromPolygon(const ColmnarDB::Types::ComplexPolygon& polygon);
};