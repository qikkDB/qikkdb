//
// Created by Jakub Veselý on 2019-01-21.
//
#include "ColumnBase.h"

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