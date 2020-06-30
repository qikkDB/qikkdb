#include "PointFactory.h"
#include "Types/Point.pb.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>

QikkDB::Types::Point PointFactory::FromWkt(std::string wktPoint)
{
    size_t openBracePos = wktPoint.find('(');
    size_t closeBracePos = wktPoint.find(')');
    if (openBracePos == std::string::npos || closeBracePos == std::string::npos)
    {
        throw std::invalid_argument("No close or open brace in WKT string.");
    }
    // remove POINT (  wkt prefix
    wktPoint.erase(0, openBracePos + 1);
    // remove )  wkt suffix
    closeBracePos = wktPoint.find(')');
    wktPoint.erase(wktPoint.begin() + closeBracePos, wktPoint.end());
    std::istringstream wktInput(wktPoint);
    QikkDB::Types::Point ret;
    float latitude, longitude;
    wktInput >> latitude >> longitude;
    if (wktInput.fail())
    {
        throw std::invalid_argument("Invalid WKT format");
    }
    ret.mutable_geopoint()->set_latitude(latitude);
    ret.mutable_geopoint()->set_longitude(longitude);
    return ret;
}

QikkDB::Types::Point PointFactory::FromLatLon(float latitude, float longitude)
{
    auto ret = QikkDB::Types::Point();
    ret.mutable_geopoint()->set_longitude(longitude);
    ret.mutable_geopoint()->set_latitude(latitude);
    return ret;
}

QikkDB::Types::Point PointFactory::FromGPUPoint(const NativeGeoPoint& point)
{
    auto ret = QikkDB::Types::Point();
    ret.mutable_geopoint()->set_longitude(point.longitude);
    ret.mutable_geopoint()->set_latitude(point.latitude);
    return ret;
}

std::string PointFactory::WktFromPoint(const QikkDB::Types::Point& point, bool fixedPrecision)
{
    std::ostringstream wktStream;
    if (fixedPrecision)
    {
        wktStream << std::fixed;
        wktStream << std::setprecision(4);
    }
    wktStream << "POINT(" << point.geopoint().latitude() << " " << point.geopoint().longitude() << ")";
    return wktStream.str();
}

std::string PointFactory::WktFromPoint(const NativeGeoPoint& point, bool fixedPrecision)
{
    std::ostringstream wktStream;
    if (fixedPrecision)
    {
        wktStream << std::fixed;
        wktStream << std::setprecision(4);
    }
    wktStream << "POINT(" << point.latitude << " " << point.longitude << ")";
    return wktStream.str();
}
