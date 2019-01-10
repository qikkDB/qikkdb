#include "ColumnBase.h"
#include <unordered_set>
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"

template<>
std::vector<int> ColumnBase<int>::NullArray(int length)
{
	return std::vector<int>(length,0);
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


template<class T>
ColumnBase<T>::ColumnBase(const std::string& name, int blockSize) :
	name_(name), blockSize_(blockSize), dataType_(typeid(T)), blocks_()
{
}

template<class T>
const BlockBase<T>& ColumnBase<T>::AddBlock()
{
	blocks_.emplace_back(*this);
	return blocks_.back();
}

template<class T>
const BlockBase<T>& ColumnBase<T>::AddBlock(const std::vector<T>& data)
{
	blocks_.emplace_back(data, *this);
	return blocks_.back();
}

template<class T>
void ColumnBase<T>::InsertData(const std::vector<T>& columnData)
{
	int startIdx = 0;
	if (blocks_.size() > 0 && !blocks_.back().IsFull())
	{
		auto & lastBlock = blocks_.back();
		if (columnData.size() <= lastBlock.EmpyBlockSpace())
		{
			lastBlock.InsertData(columnData);
			return;
		}
		int emptySpace = lastBlock.EmptyBlockSpace();
		lastBlock.InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
		startIdx += emptySpace;
	}

	while (startIdx < columnData.size())
	{
		int toCopy = columnData.size() - startIdx < blockSize_
			? columnData.size() - startIdx
			: blockSize_;
		AddBlock(std::vector<T>(columnData.cbegin() + startIdx, columnData.cbegin() + startIdx + toCopy));
		startIdx += toCopy;
	}
}

template<class T>
std::vector<T> ColumnBase<T>::GetUniqueBuckets() const
{
	std::unordered_set<T> dataSet;
	auto floatBlocks = GetBlocksList();
	for(const auto & block : floatBlocks)
	{
		for(const auto & dataPoint : block.GetData())
		{
			dataSet.insert(dataPoint);
		}
	}
	return std::vector<T>(dataSet.cbegin(),dataSet.cend());
}

template<class T>
void ColumnBase<T>::InsertNullData(int length)
{
	InsertData(NullArray(length));
}
