#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"

#include <numeric>
#include <algorithm>
#include <cmath>

template<>
void BlockBase<int32_t>::setBlockStatistics()
{
	min_ = *std::min_element(data_.begin(), data_.end());
	max_ = *std::max_element(data_.begin(), data_.end());
	avg_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0)/data_.size();
	sum_ = std::accumulate(data_.begin(), data_.end(), 0);
}

template<>
void BlockBase<int64_t>::setBlockStatistics()
{
	min_ = *std::min_element(data_.begin(), data_.end());
	max_ = *std::max_element(data_.begin(), data_.end());
	avg_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0) / data_.size();
	sum_ = std::accumulate(data_.begin(), data_.end(), 0);
}

template<>
void BlockBase<float>::setBlockStatistics()
{
	min_ = *std::min_element(data_.begin(), data_.end());
	max_ = *std::max_element(data_.begin(), data_.end());
	avg_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0) / data_.size();
	sum_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0);
}

template<>
void BlockBase<double>::setBlockStatistics()
{
	min_ = *std::min_element(data_.begin(), data_.end());
	max_ = *std::max_element(data_.begin(), data_.end());
	avg_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0) / data_.size();
	sum_ = std::accumulate(data_.begin(), data_.end(), (double) 0.0);
}

template<>
void BlockBase<ColmnarDB::Types::Point>::setBlockStatistics()
{
	min_ = PointFactory::FromWkt("POINT(0 0)");
	max_ = PointFactory::FromWkt("POINT(0 0)");
	avg_ = (float) 0.0;
	sum_ = PointFactory::FromWkt("POINT(0 0)");
}

template<>
void BlockBase<ColmnarDB::Types::ComplexPolygon>::setBlockStatistics()
{
	min_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
	max_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
	avg_ = (float) 0.0;
	sum_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
}

template<>
void BlockBase<std::string>::setBlockStatistics()
{
	min_ = "";
	max_ = "";
	avg_ = (float) 0.0;
	sum_ = "";
}

template<>
void BlockBase<int8_t>::setBlockStatistics()
{
	min_ = *std::min_element(data_.begin(), data_.end());
	max_ = *std::max_element(data_.begin(), data_.end());
	avg_ = std::accumulate(data_.begin(), data_.end(), (float) 0.0) / data_.size();
	sum_ = std::accumulate(data_.begin(), data_.end(), 0);
}