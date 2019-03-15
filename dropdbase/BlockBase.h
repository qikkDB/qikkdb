#pragma once
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Compression/Compression.h"

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
	size_t compressedSize_;
	size_t capacity_;
	std::unique_ptr<T[]> data_;

	bool isCompressed_;

public:
	BlockBase(const std::vector<T>& data, ColumnBase<T>& column, bool isCompressed = false) :
		column_(column), size_(0), isCompressed_(isCompressed)
	{
		capacity_ = (isCompressed) ? data.size() : column_.GetBlockSize();
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
		if (isCompressed_)
			return false;
		else
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

	bool IsCompressed() const
	{
		return isCompressed_;
	}
	
	size_t GetCompressedSize() const
	{
		return compressedSize_;
	}

	ColumnBase<T>& GetColumn()
	{
		return column_;
	}
	
	void CompressData() 
	{
		if (isCompressed_)
			return;

		bool compressedSuccessfully = false;
		std::vector<T> dataCompressed;
		Compression::Compress(column_.GetColumnType(), data_.get(), size_, dataCompressed, compressedSuccessfully);
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
		Compression::Decompress(column_.GetColumnType(), data_.get(), size_, dataDecompressed, decompressedSuccessfully);
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

	~BlockBase()
	{
		GPUMemory::hostUnregister(data_.get());
	}

	BlockBase(const BlockBase&) = delete;
	BlockBase& operator=(const BlockBase&) = delete;
};
