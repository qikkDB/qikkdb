#pragma once
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

template <class T>
class ColumnBase;

/// <summary>
/// The main class representing block of data. Holds data of type T.
/// </summary>
/// <typeparam name="T">Data type of block.</typeparam>
template<class T>
class BlockBase
{
private:
	T min_ = std::numeric_limits<T>::lowest();
	T max_ = std::numeric_limits<T>::max();
	float avg_ = 0.0;
	T sum_ = T{};
	int32_t groupId_ = -1; //index for group of blocks - binary index
	
	void setBlockStatistics();

    ColumnBase<T>& column_;
    size_t size_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;

public:
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class filled with data.
	/// </summary>
	/// <param name="data">Data which will fill up the block.</param>
	/// <param name="column">Column that will hold this new block.</param>
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

	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class without data.
	/// </summary>
	/// <param name="column">Column that will hold this new empty block.</param>
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
	
	int32_t GetGroupId()
	{
		return groupId_;
	}

    T* const GetData()
    {
        return data_.get();
    }

    size_t GetSize() const
    {
        return size_;
    }

	/// <summary>
	/// Find out the amount of empty space in current block.
	/// </summary>
	/// <returns>Block space that is not filled with data.</returns>
    int EmptyBlockSpace() const
    {
        return capacity_ - size_;
    }

    int BlockCapacity() const
    {
        return capacity_;
    }

	/// <summary>
	/// Find out wheather current block is completely filled with data.
	/// </summary>
	/// <returns>Returns true if block is full. If block is not full, returns false.</returns>
    bool IsFull() const
    {
        return EmptyBlockSpace() == 0;
    }

    std::tuple<int, int, bool>
    FindIndexAndRange(int indexInBlock, int range, const T& data)
    {
        int newRange = 0;
        int newIndexInBlock = indexInBlock;
        bool reachEnd = false;

		// flag if some data in block equals to input data is found
		bool found = false;
		// flag if for loop is broken because of some conditions
        bool inRange = false;

		if (size_ == 0)
        {
            newIndexInBlock = 0;
            newRange = 0;
            reachEnd = true;
        }

		else
        {
            for (int i = indexInBlock; i <= indexInBlock + range; i++)
            {
                // index out of block
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
                    // if first checked value is greater than data
                    if (!found)
                    {
                        newIndexInBlock = i;
                        inRange = true;
                        found = true;
                        break;
                    }

                    // last suitable value
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

            // if whole for loop was executed
            if (!inRange)
            {
                if (found)
                {
                    newRange = indexInBlock + range - newIndexInBlock;
                }
                else
                {
                    // if suitable value was not found, index at end is chosen as place to insert
                    newIndexInBlock = indexInBlock + range;
                }
            }
        }

		return std::make_tuple(newIndexInBlock, newRange, reachEnd);
    }

    /// <summary>
	/// Insert data into the current block.
	/// </summary>
	/// <param name="data">Data to be inserted.</param>
	/// <exception cref="std::length_error">Attempted to insert data larger than remaining block size.</exception>
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
        int filledBlockSpace = column_.GetBlockSize() - EmptyBlockSpace();

        if (EmptyBlockSpace() == 0)
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }

        else if (index < filledBlockSpace)
        {
            for (int j = filledBlockSpace - 1; j >= index; j--)
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
