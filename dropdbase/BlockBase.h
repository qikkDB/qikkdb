#pragma once
#include "IBlock.h"
#include <stdexcept>

template<class T>
class ColumnBase;

template<class T>
class BlockBase :
	public IBlock<T>
{
private:
	std::vector<T> data_;
	ColumnBase<T>& column_;
public:
	BlockBase(const std::vector<T>& data, ColumnBase<T>& column) :
		column_(column), data_(data)
	{
		if (column_.GetBlockSize() - data.size() < 0)
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}
		data_.reserve(column_.GetBlockSize());
	}

	explicit BlockBase(ColumnBase<T>& column) :
		column_(column), data_()
	{
		data_.reserve(column_.GetBlockSize());
	}

	virtual std::vector<T>& GetData() override
	{
		return data_;
	}
	
	virtual int EmptyBlockSpace() const override
	{
		return column_.GetBlockSize() - data_.size();
	}

	virtual bool IsFull() const override
	{
		return EmptyBlockSpace() == 0;
	}

	virtual void InsertData(const std::vector<T>& data) override
	{
		if (EmptyBlockSpace() - data.size() < 0)
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}
		data_.insert(data_.end(), data.cbegin(), data.cend());
	}
};

