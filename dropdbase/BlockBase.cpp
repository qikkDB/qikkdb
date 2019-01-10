#include "BlockBase.h"
#include <exception>

template<class T>
BlockBase<T>::BlockBase(const std::vector<T>& data, const ColumnBase<T>& column) :
	column_(column), data_(data)
{
	if (column_.GetBlockSize() - data.size() < 0)
	{
		throw std::length_error("Attempted to insert data larger than remaining block size");
	}
	data_.reserve(column_.GetBlockSize());
}

template<class T>
BlockBase<T>::BlockBase(const ColumnBase<T>& column) :
	column_(column), data_()
{
	data_.reserve(column_.GetBlockSize());
}

template<class T>
std::vector<T> BlockBase<T>::GetData() const
{
	return data_;
}

template<class T>
int BlockBase<T>::EmptyBlockSpace() const
{
	return column_.GetBlockSize() - data_.size();
}

template<class T>
bool BlockBase<T>::IsFull() const
{
	return EmptyBlockSpace() == 0;
}

template<class T>
void BlockBase<T>::InsertData(const std::vector<T>& data)
{
	if (EmptyBlockSpace() - data.size() < 0)
	{
		throw std::length_error("Attempted to insert data larger than remaining block size");
	}
	data_.insert(data_.end(), data.cbegin(), data.cend());
}
