#pragma once
#include <string>
#include "NativeGeoPoint.h"

namespace QikkDB
{
namespace Types
{
class Point;
}
} // namespace QikkDB

/// <summary>
/// The class for creating points from different sources and to create well-known-text from point.
/// </summary>
class PointFactory
{
private:
    PointFactory(){};

public:
    static QikkDB::Types::Point FromWkt(std::string wktPoint);
    static QikkDB::Types::Point FromLatLon(float latitude, float longitude);
    static QikkDB::Types::Point FromGPUPoint(const NativeGeoPoint& point);
    static std::string WktFromPoint(const QikkDB::Types::Point& point, bool fixedPrecision = false);
    static std::string WktFromPoint(const NativeGeoPoint& point, bool fixedPrecision = false);
};
