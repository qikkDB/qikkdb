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
	T min_ = std::numeric_limits<T>::lowest();
	T max_ = std::numeric_limits<T>::max();
	float avg_ = 0.0;
	T sum_ = T{0};
	int32_t groupId_ = -1; //index for group of blocks - binary index

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

	int32_t GetGroupId()
	{
		return groupId_;
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
	
	~BlockBase()
	{
		GPUMemory::hostUnregister(data_.get());
	}

	BlockBase(const BlockBase&) = delete;
	BlockBase& operator=(const BlockBase&) = delete;
};
