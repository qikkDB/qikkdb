#pragma once
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

template <class T>
class ColumnBase;

template <class T>
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
    BlockBase(const std::vector<T>& data, ColumnBase<T>& column)
    : column_(column), size_(0), capacity_(column_.GetBlockSize()), data_(new T[capacity_])
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

    explicit BlockBase(ColumnBase<T>& column)
    : column_(column), size_(0), capacity_(column_.GetBlockSize()), data_(new T[capacity_])
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

    T* const GetData()
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

    int BlockCapacity() const
    {
        return capacity_;
    }

    bool IsFull() const
    {
        return EmptyBlockSpace() == 0;
    }

    std::tuple<int, int, bool>
    FindIndexAndRange(int indexInBlock, int range, const T& data)
    {
        int newRange = 1;
        int newIndexInBlock = indexInBlock;
        bool reachEnd = false;

		// flag if some data in block equals to input data is found
		bool found = false;
		// flag if for loop is broken because of some conditions
        bool inRange = false;
        for (int i = indexInBlock; i < indexInBlock + range; i++)
        {
			//index out of block
            if (i >= size_)
            {
                reachEnd = true;
                if (found)
                {
                    newRange = i - newIndexInBlock;
				}
				else
				{
                    newIndexInBlock = size_;
                }
                inRange = true;
                break;
			}

			if (data_[i] > data)
			{
				//if first checked value is greater than data
				if (!found)
				{
                    newIndexInBlock = i;
                    inRange = true;
                    break;
				}

				//last suitable value
                newRange = i - newIndexInBlock;
                inRange = true;
                break;
			}

			if (data_[i] == data)
			{
				if (!found)
				{
                    newIndexInBlock = i;
					found = true;
				}
			}
		}

		//if whole for loop was executed
		if (!inRange)
		{
            if (found)
            {
                newRange = indexInBlock + range - newIndexInBlock;
			}
            else
            {
				//if suitable value was not found, index at end is chosen as place to insert
				newIndexInBlock = indexInBlock + range;
			}
		}

		return std::make_tuple(newIndexInBlock, newRange, reachEnd);
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
