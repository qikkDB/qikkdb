//
// Created by Jakub Veselý on 2019-01-21.
//
#include "ColumnBase.h"
#include <numeric>
#include <algorithm>
#include <cmath>

template<>
std::vector<int> ColumnBase<int>::NullArray(int length)
{
	return std::vector<int>(length, 0);
}

template<>
std::vector<float> ColumnBase<float>::NullArray(int length)
{
	return std::vector<float>(length, 0);
}

template<>
std::vector<long long> ColumnBase<long long>::NullArray(int length)
{
	return std::vector<long long>(length, 0);
}

template<>
std::vector<double> ColumnBase<double>::NullArray(int length)
{
	return std::vector<double>(length, 0);
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
		numOfDataInBlocks.push_back(block->GetData().size());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), 0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(),numOfDataInBlocks.end(),0);
}

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
		numOfDataInBlocks.push_back(block->GetData().size());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), 0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), 0);
}

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
		numOfDataInBlocks.push_back(block->GetData().size());
	}

	min_ = *std::min_element(mins.begin(), mins.end());
	max_ = *std::max_element(maxs.begin(), maxs.end());
	sum_ = std::accumulate(sums.begin(), sums.end(), (float) 0.0);
	avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float) 0.0);
}

void ColumnBase<double>::setColumnStatistics()
{
}

void ColumnBase<ColmnarDB::Types::Point>::setColumnStatistics()
{
}

void ColumnBase<ColmnarDB::Types::ComplexPolygon>::setColumnStatistics()
{
}

void ColumnBase<std::string>::setColumnStatistics()
{
}

void ColumnBase<bool>::setColumnStatistics()
{
}