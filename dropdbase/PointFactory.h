#pragma once
#include <string>
#include "NativeGeoPoint.h"

namespace ColmnarDB
{
	namespace Types
	{
		class Point;
	}
}

/// <summary>
/// The class for creating points from different sources and to create well-known-text from point.
/// </summary>
class PointFactory
{
private:
	PointFactory() {};
public:
	static ColmnarDB::Types::Point FromWkt(std::string wktPoint);
	static ColmnarDB::Types::Point FromLatLon(float latitude, float longitude);
	static ColmnarDB::Types::Point FromGPUPoint(const NativeGeoPoint& point);
	static std::string WktFromPoint(const ColmnarDB::Types::Point& point, bool fixedPrecision = false);
	static std::string WktFromPoint(const NativeGeoPoint& point, bool fixedPrecision = false);
};

