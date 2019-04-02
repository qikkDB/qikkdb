#include "PointFactory.h"
#include <sstream>
#include <stdexcept>

ColmnarDB::Types::Point PointFactory::FromWkt(std::string wktPoint)
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
	ColmnarDB::Types::Point ret;
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

ColmnarDB::Types::Point PointFactory::FromLatLon(float latitude, float longitude)
{
	auto ret = ColmnarDB::Types::Point();
	ret.mutable_geopoint()->set_longitude(longitude);
	ret.mutable_geopoint()->set_latitude(latitude);
	return ret;
}

ColmnarDB::Types::Point PointFactory::FromGPUPoint(const NativeGeoPoint& point)
{
	auto ret = ColmnarDB::Types::Point();
	ret.mutable_geopoint()->set_longitude(point.longitude);
	ret.mutable_geopoint()->set_latitude(point.latitude);
	return ret;
}

std::string PointFactory::WktFromPoint(const ColmnarDB::Types::Point & point)
{
	std::ostringstream wktStream;
	wktStream << "POINT(" << point.geopoint().latitude() << " " << point.geopoint().longitude() << ")";
	return wktStream.str();
}

std::string PointFactory::WktFromPoint(const NativeGeoPoint & point)
{
	std::ostringstream wktStream;
	wktStream << "POINT(" << point.latitude << " " << point.longitude << ")";
	return wktStream.str();
}
