#pragma once
#include <string>
#include <typeinfo>
#include <vector>

#include "IBlock.h"
#include "BlockBase.h"

template<class T>
class ColumnBase
{
private:
	std::string name_;
	int blockSize_;
	std::type_info dataType_;
	std::vector<IBlock<T>> blocks_;

	std::vector<T> NullArray(int length);
public:
	ColumnBase(const std::string& name, int blockSize)	:
		name_(name), blockSize_(blockSize), dataType_(typeid(T)), blocks_()
	{
	}

	inline int GetBlockSize() const { return blockSize_; };
	inline const std::string& GetName() const { return name_; };

	/// <summary>
	/// Blocks getter
	/// </summary>
	/// <returns>List of blocks in current column</returns>
	const std::vector<IBlock<T>>& GetBlocksList() const
	{
		return blocks_;
	};

	/// <summary>
	/// Add new block in column
	/// </summary>
	/// <returns>Last block of column</returns>
	const BlockBase<T>& AddBlock()
	{
		blocks_.emplace_back(*this);
		return blocks_.back();
	}

	/// <summary>
	/// Add new block with proper data into column
	/// </summary>
	/// <param name="data">Data to be inserted</param>
	/// <returns>Last block of column</returns>
	const BlockBase<T>& AddBlock(const std::vector<T>& data)
	{
		blocks_.emplace_back(data, *this);
		return blocks_.back();
	}


	/// <summary>
	/// Insert data into column considering empty space of last block and maximum size of blocks
	/// </summary>
	/// <param name="columnData">Data to be inserted</param>
	void InsertData(const std::vector<T>& columnData)
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

	/// <summary>
	/// Get all unique values for this column
	/// </summary>
	/// <returns>Array of unique values</returns>
	std::vector<T> GetUniqueBuckets() const
	{
		std::unordered_set<T> dataSet;
		auto floatBlocks = GetBlocksList();
		for (const auto & block : floatBlocks)
		{
			for (const auto & dataPoint : block.GetData())
			{
				dataSet.insert(dataPoint);
			}
		}
		return std::vector<T>(dataSet.cbegin(), dataSet.cend());
	}

	/// <summary>
	/// Insert null data into column
	/// </summary>
	/// <param name="length">Length of inserted data</param>
	void InsertNullData(int length)
	{
		InsertData(NullArray(length));
	}

	/// <summary>
	/// Returns type of ColumnBase
	/// </summary>
	/// <returns>Type of current column</returns>
	std::type_info GetColumnType() const
	{
		return dataType_;
	};
};

template<>
std::vector<int> ColumnBase<int>::NullArray(int length)
{
	return std::vector<int>(length, 0);
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