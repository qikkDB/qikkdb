#pragma once
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
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

	ColumnBase<T>& column_;
	size_t size_;
	size_t capacity_;
	std::unique_ptr<T[]> data_;
public:
	BlockBase(const std::vector<T>& data, ColumnBase<T>& column) :
		column_(column), size_(0), capacity_(column_.GetBlockSize()), data_(new T[capacity_])
	{
		if (column_.GetBlockSize() < data.size())
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}
		GPUMemory::hostPin(data_.get(), capacity_);
		std::copy(data.begin(), data.end(), data_.get());
		size_ = data.size();
		setBlockStatistics();
	}

	explicit BlockBase(ColumnBase<T>& column) :
		column_(column), size_(0), capacity_(column_.GetBlockSize()), data_(new T[capacity_])
	{
		GPUMemory::hostPin(data_.get(), capacity_);
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

	T * const GetData()
	{
		return data_.get();
	}

	size_t GetSize() const
	{
		return size_;
	}

	int EmptyBlockSpace() const
	{
		return capacity_ - size_;
	}

	bool IsFull() const
	{
		return EmptyBlockSpace() == 0;
	}

	std::tuple<int,int,bool> FindIndexAndRange(const T& data)
	{
		int index;
		int fullBlockSpace = column_.GetBlockSize() - EmptyBlockSpace();
		int range = 0;
		bool reachEnd = false;

		if (EmptyBlockSpace() == 0)
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}

		else if (fullBlockSpace == 0)
		{
			range = 0;
			index = 0;
			reachEnd = true;
		}

		else if (data_[fullBlockSpace-1] < data)
		{
			index = fullBlockSpace;
			range = 0;
			reachEnd = true;
		}

		else if (data_[0] >= data)
		{
			index = 0;
			for (int i = 0; i < (fullBlockSpace - 1); i++)
			{
				if (data_[i] == data) {
					range++;
				}
				else
				{
					break;
				}
			}
			if (data_[fullBlockSpace] == data) {
				reachEnd = true;
			}
			else
			{
				reachEnd = false;
			}
		}

		else
		{
			for (int i = 0; i < (fullBlockSpace - 1); i++)
			{
				if (data_[i] < data && data_[i + 1] >= data)
				{
					index = i + 1;
					for (int j = i + 1; j < (fullBlockSpace - 1); j++)
					{
						if (data_[j] == data) {
							range++;
						}
						else
						{
							break;
						}
					}
					break;
				}
				if (data_[fullBlockSpace] == data) {
					reachEnd = true;
				}
				else
				{
					reachEnd = false;
				}
			}
		}
		return std::make_tuple(index,range,reachEnd);
	}

	void InsertData(const std::vector<T>& data)
	{
		if (EmptyBlockSpace() < data.size())
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}
		std::copy(data.begin(), data.end(), data_.get() + size_);
		size_ += data.size();
		setBlockStatistics();
	}

	void InsertDataOnSpecificPosition(int index, const T& data)
	{
		int fullBlockSpace = column_.GetBlockSize() - EmptyBlockSpace();

		if (EmptyBlockSpace() == 0)
		{
			throw std::length_error("Attempted to insert data larger than remaining block size");
		}

		else if (index < fullBlockSpace)
		{
			for (int j = fullBlockSpace - 1; j >= index; j--)
			{
				data_[j + 1] = data_[j];
			}
		}
		data_[index] = data;
		size_++;
		setBlockStatistics();
	}
	
	~BlockBase()
	{
		GPUMemory::hostUnregister(data_.get());
	}

	BlockBase(const BlockBase&) = delete;
	BlockBase& operator=(const BlockBase&) = delete;
};
