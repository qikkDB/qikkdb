#pragma once
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Compression/Compression.h"

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
	size_t compressedSize_;
	size_t capacity_;
	std::unique_ptr<T[]> data_;

	bool isCompressed_;

public:
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class filled with data.
	/// </summary>
	/// <param name="data">Data which will fill up the block.</param>
	/// <param name="column">Column that will hold this new block.</param>
	BlockBase(const std::vector<T>& data, ColumnBase<T>& column, bool isCompressed = false) :
		column_(column), size_(0), isCompressed_(isCompressed)
	{
		capacity_ = (isCompressed) ? data.size() : column.GetBlockSize();
		data_ = std::unique_ptr<T[]>(new T[capacity_]);

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
	explicit BlockBase(ColumnBase<T>& column) :
		column_(column), size_(0), capacity_(column_.GetBlockSize()), data_(new T[capacity_])
	{
		GPUMemory::hostPin(data_.get(), capacity_);
		isCompressed_ = false;
	}

	void setBlockStatistics(T min, T max, float avg, T sum)
	{
		min_ = min;
		max_ = max;
		avg_ = avg;
		sum_ = sum;
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

    T * GetData() 
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
	/// <summary>
	bool IsFull() const
	{
		if (isCompressed_)
			return true;
		else
			return EmptyBlockSpace() == 0;
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

	bool IsCompressed() const
	{
		return isCompressed_;
	}
	
	size_t GetCompressedSize() const
	{
		return compressedSize_;
	}
	
	void CompressData() 
	{
		if (isCompressed_)
			return;

		bool compressedSuccessfully = false;
		std::vector<T> dataCompressed;
		Compression::Compress(column_.GetColumnType(), data_.get(), size_, dataCompressed, min_, max_, compressedSuccessfully);
		if (compressedSuccessfully)
		{
			GPUMemory::hostUnregister(data_.get());
			
			capacity_ = dataCompressed.size();
			size_ = dataCompressed.size();
			data_.release();
			data_ = std::unique_ptr<T[]>(new T[capacity_]);
			
			GPUMemory::hostPin(data_.get(), capacity_);
			std::copy(dataCompressed.begin(), dataCompressed.end(), data_.get());
			
			isCompressed_ = true;
		}
		
	}

	void DecompressData()
	{
		if (!isCompressed_)
			return;

		bool decompressedSuccessfully = false;
		std::vector<T> dataDecompressed;		
		int64_t uncompressedElementsCount = Compression::GetUncompressedDataElementsCount(data_.get());
		int64_t compressionBlocksCount = Compression::GetCompressionBlocksCount(data_.get());
		Compression::Decompress(column_.GetColumnType(), data_.get(), size_, dataDecompressed, uncompressedElementsCount, compressionBlocksCount, min_, max_, decompressedSuccessfully);
		if (decompressedSuccessfully)
		{
			GPUMemory::hostUnregister(data_.get());

			capacity_ = column_.GetBlockSize();
			size_ = column_.GetBlockSize();
			data_.release();
			data_ = std::unique_ptr<T[]>(new T[capacity_]);

			GPUMemory::hostPin(data_.get(), capacity_);
			std::copy(dataDecompressed.begin(), dataDecompressed.end(), data_.get());

			isCompressed_ = false;
		}

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
           /* for (int j = filledBlockSpace - 1; j >= index; j--)
            {
                data_[j + 1] = data_[j];
            }*/

			std::move(data_.get() + index, data_.get() + size_, data_.get() + index + 1);
        }
        data_[index] = data;
        size_++;
        //setBlockStatistics();
    }

    ~BlockBase()
    {
        GPUMemory::hostUnregister(data_.get());
    }

    BlockBase(const BlockBase&) = delete;
    BlockBase& operator=(const BlockBase&) = delete;
};
