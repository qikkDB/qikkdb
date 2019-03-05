#include "ColumnBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"

#include <numeric>
#include <algorithm>
#include <cmath>

template<>
std::vector<int32_t> ColumnBase<int32_t>::NullArray(int length)
{
	return std::vector<int32_t>(length, 0);
}

template<>
std::vector<float> ColumnBase<float>::NullArray(int length)
{
	return std::vector<float>(length, 0);
}

template<>
std::vector<int64_t> ColumnBase<int64_t>::NullArray(int length)
{
	return std::vector<int64_t>(length, 0);
}

template<>
std::vector<double> ColumnBase<double>::NullArray(int length)
{
	return std::vector<double>(length, 0);
}

template<>
std::vector<int8_t> ColumnBase<int8_t>::NullArray(int length)
{
	return std::vector<int8_t>(length, 0);
}

template<>
std::vector<std::string> ColumnBase<std::string>::NullArray(int length)
{
	return std::vector<std::string>(length, "");
}

template<>
std::vector<ColmnarDB::Types::Point> ColumnBase<ColmnarDB::Types::Point>::NullArray(int length)
{
	return std::vector<ColmnarDB::Types::Point>(length, ColmnarDB::Types::Point());
}

template<>
std::vector<ColmnarDB::Types::ComplexPolygon> ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(int length)
{
	return std::vector<ColmnarDB::Types::ComplexPolygon>(length, ColmnarDB::Types::ComplexPolygon());
}

template<>
void ColumnBase<int32_t>::setColumnStatistics()
{
	std::vector<int32_t> mins;
	std::vector<int32_t> maxs;
	std::vector<int32_t> sums;

	std::vector<int64_t> numOfDataInBlocks;

	for (auto& block : blocks_)
	{
		mins.push_back(block->GetMin());
		maxs.push_back(block->GetMax());
		sums.push_back(block->GetSum());
		numOfDataInBlocks.push_back(block->GetSize());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), 0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(),numOfDataInBlocks.end(), (float) 0.0);
}

template<>
void ColumnBase<int64_t>::setColumnStatistics()
{
	std::vector<int64_t> mins;
	std::vector<int64_t> maxs;
	std::vector<int64_t> sums;

	std::vector<int64_t> numOfDataInBlocks;

	for (auto& block : blocks_)
	{
		mins.push_back(block->GetMin());
		maxs.push_back(block->GetMax());
		sums.push_back(block->GetSum());
		numOfDataInBlocks.push_back(block->GetSize());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), 0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float) 0.0);
}

template<>
void ColumnBase<float>::setColumnStatistics()
{
	std::vector<float> mins;
	std::vector<float> maxs;
	std::vector<float> sums;

	std::vector<int64_t> numOfDataInBlocks;

	for (auto& block : blocks_)
	{
		mins.push_back(block->GetMin());
		maxs.push_back(block->GetMax());
		sums.push_back(block->GetSum());
		numOfDataInBlocks.push_back(block->GetSize());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), (float) 0.0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float) 0.0);
}

template<>
void ColumnBase<double>::setColumnStatistics()
{
	std::vector<double> mins;
	std::vector<double> maxs;
	std::vector<double> sums;

	std::vector<int64_t> numOfDataInBlocks;

	for (auto& block : blocks_)
	{
		mins.push_back(block->GetMin());
		maxs.push_back(block->GetMax());
		sums.push_back(block->GetSum());
		numOfDataInBlocks.push_back(block->GetSize());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), (double) 0.0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float) 0.0);
}

template<>
void ColumnBase<ColmnarDB::Types::Point>::setColumnStatistics()
{
	min_ = PointFactory::FromWkt("POINT(0 0)");
	max_ = PointFactory::FromWkt("POINT(0 0)");
	avg_ = (float) 0.0;
	sum_ = PointFactory::FromWkt("POINT(0 0)");
}

template<>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::setColumnStatistics()
{
	min_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
	max_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
	avg_ = (float) 0.0;
	sum_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
}

template<>
void ColumnBase<std::string>::setColumnStatistics()
{
	min_ = "";
	max_ = "";
	avg_ = (float) 0.0;
	sum_ = "";
}

template<>
void ColumnBase<int8_t>::setColumnStatistics()
{
	std::vector<int8_t> mins;
	std::vector<int8_t> maxs;
	std::vector<int8_t> sums;

	std::vector<int64_t> numOfDataInBlocks;

	for (auto& block : blocks_)
	{
		mins.push_back(block->GetMin());
		maxs.push_back(block->GetMax());
		sums.push_back(block->GetSum());
		numOfDataInBlocks.push_back(block->GetSize());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), 0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float) 0.0);
}
