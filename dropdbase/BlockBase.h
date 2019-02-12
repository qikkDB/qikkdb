#pragma once
#include <stdexcept>
#include <vector>
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"

template<class T>
class ColumnBase;

template<class T>
class BlockBase
{
private:
	T min_;
	T max_;
	float avg_;
	T sum_;

	void setBlockStatistics();

	std::vector<T> data_;
	ColumnBase<T>& column_;
public:
	BlockBase(const std::vector<T>& data, ColumnBase<T>& column) :
		column_(column), data_(data)
	{
		if (column_.GetBlockSize() < data.size())
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}
		data_.reserve(column_.GetBlockSize());
		GPUMemory::hostPin(data_.data(), data_.size());
		setBlockStatistics();
	}

	explicit BlockBase(ColumnBase<T>& column) :
		column_(column), data_()
	{
		data_.reserve(column_.GetBlockSize());
		GPUMemory::hostPin(data_.data(), data_.size());
	}

	T GetMax()
	{
		return max_;
	}

	T GetMin()
	{
		return min_;
	}

	float GetAvg()
	{
		return avg_;
	}

	T GetSum()
	{
		return sum_;
	}

	std::vector<T>& GetData()
	{
		return data_;
	}

	int EmptyBlockSpace() const
	{
		return column_.GetBlockSize() - data_.size();
	}

	bool IsFull() const
	{
		return EmptyBlockSpace() == 0;
	}

	int32_t InsertData(const T& data)
	{
		int index;
		int fullBlockSpace = column_.GetBlockSize() - EmptyBlockSpace();

		if (EmptyBlockSpace() == 0)
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}

		else if (fullBlockSpace == 0)
		{
			data_[0] = data;
			index = 0;
		}

		else if (data_[fullBlockSpace] < data)
		{
			data_[fullBlockSpace + 1] = data;
			index = fullBlockSpace + 1;
		}

		else
		{
			for (int i = 0; i < (fullBlockSpace - 1); i++)
			{
				if (data_[i] < data && data_[i + 1] >= data)
				{
					for (j = fullBlockSpace; j > i; j--)
					{
						data_[j + 1] = data_[j];
					}
					data_[i + 1] = data;
					index = i + 1;
					break;
				}
			}
		}
		setBlockStatistics();
		return index;
	}
	
	~BlockBase()
	{
		GPUMemory::hostUnregister(data_.data());
	}

	BlockBase(const BlockBase&) = delete;
	BlockBase& operator=(const BlockBase&) = delete;
};
