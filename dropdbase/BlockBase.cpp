#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include <numeric>
#include <algorithm>
#include <cmath>

template<>
void BlockBase<int32_t>::setBlockStatistics()
{
	if (isCompressed_)
	{
		return;
	}

	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0)/ size_;
	sum_ = std::accumulate(data_.get(), data_.get() + size_, 0);
}

template<>
void BlockBase<int64_t>::setBlockStatistics()
{
	if (isCompressed_)
	{
		return;
	}

	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0) / size_;
	sum_ = std::accumulate(data_.get(), data_.get() + size_, 0);
}

template<>
void BlockBase<float>::setBlockStatistics()
{
	if (isCompressed_)
	{
		return;
	}

	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0) / size_;
	sum_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0);
}

template<>
void BlockBase<double>::setBlockStatistics()
{
	if (isCompressed_)
	{
		return;
	}

	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0) / size_;
	sum_ = std::accumulate(data_.get(), data_.get() + size_, (double) 0.0);
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
	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = (float) 0.0;
	sum_ = "";
}

template<>
void BlockBase<int8_t>::setBlockStatistics()
{
	if (isCompressed_)
	{
		return;
	}

	min_ = *std::min_element(data_.get(), data_.get() + size_);
	max_ = *std::max_element(data_.get(), data_.get() + size_);
	avg_ = std::accumulate(data_.get(), data_.get() + size_, (float) 0.0) / size_;
	sum_ = std::accumulate(data_.get(), data_.get() + size_, 0);
}