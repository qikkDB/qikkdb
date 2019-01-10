#pragma once
#include "Types/Point.pb.h"
#include <string>
#include "NativeGeoPoint.h"
class PointFactory
{
private:
	PointFactory() {};
public:
	static ColmnarDB::Types::Point FromWkt(std::string wktPoint);
	static ColmnarDB::Types::Point FromLatLon(float latitude, float longitude);
	static ColmnarDB::Types::Point FromGPUPoint(const NativeGeoPoint& point);
	static std::string WktFromPoint(const ColmnarDB::Types::Point& point);
};

