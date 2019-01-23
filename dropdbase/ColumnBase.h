#pragma once
#include <string>
#include <typeinfo>
#include <vector>

#include "IBlock.h"
#include "BlockBase.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "IColumn.h"

template<class T>
class ColumnBase : public IColumn
{
private:
	std::string name_;
	int blockSize_;
	std::vector<std::unique_ptr<IBlock<T>>> blocks_;

	std::vector<T> NullArray(int length);
public:
	ColumnBase(const std::string& name, int blockSize) :
		name_(name), blockSize_(blockSize), blocks_()
	{
	}

	inline int GetBlockSize() const { return blockSize_; };

	virtual const std::string& GetName() const override
	{
		return name_;
	}

	/// <summary>
	/// Blocks getter
	/// </summary>
	/// <returns>List of blocks in current column</returns>
	const std::vector<std::unique_ptr<IBlock<T>>>& GetBlocksList() const
	{
		return blocks_;
	};

	/// <summary>
	/// Add new block in column
	/// </summary>
	/// <returns>Last block of column</returns>
	const BlockBase<T>& AddBlock()
	{
		blocks_.push_back(std::make_unique<BlockBase<T>>(*this));
		return *(dynamic_cast<BlockBase<T>*>(blocks_.back().get()));
	}

	/// <summary>
	/// Add new block with proper data into column
	/// </summary>
	/// <param name="data">Data to be inserted</param>
	/// <returns>Last block of column</returns>
	const BlockBase<T>& AddBlock(const std::vector<T>& data)
	{
		blocks_.push_back(std::make_unique<BlockBase<T>>(data, *this));
		return *(dynamic_cast<BlockBase<T>*>(blocks_.back().get()));
	}


	/// <summary>
	/// Insert data into column considering empty space of last block and maximum size of blocks
	/// </summary>
	/// <param name="columnData">Data to be inserted</param>
	void InsertData(const std::vector<T>& columnData)
	{
		int startIdx = 0;
		if (blocks_.size() > 0 && !blocks_.back()->IsFull())
		{
			auto & lastBlock = blocks_.back();
			if (columnData.size() <= lastBlock->EmptyBlockSpace())
			{
				lastBlock->InsertData(columnData);
				return;
			}
			int emptySpace = lastBlock->EmptyBlockSpace();
			lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
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
			for (const auto & dataPoint : block->GetData())
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
	virtual DataType GetColumnType() const override
	{
		typedef typename std::conditional<std::is_same<T, int>::value, std::integral_constant<DataType, COLUMN_INT>,
			typename std::conditional<std::is_same<T, int64_t>::value, std::integral_constant<DataType, COLUMN_LONG>,
			typename std::conditional<std::is_same<T, float>::value, std::integral_constant<DataType, COLUMN_FLOAT>,
			typename std::conditional<std::is_same<T, double>::value, std::integral_constant<DataType, COLUMN_DOUBLE>,
			typename std::conditional<std::is_same<T, ColmnarDB::Types::Point>::value, std::integral_constant<DataType, COLUMN_POINT>,
			typename std::conditional<std::is_same<T, ColmnarDB::Types::ComplexPolygon>::value, std::integral_constant<DataType, COLUMN_POLYGON>,
			typename std::conditional<std::is_same<T, std::string>::value, std::integral_constant<DataType, COLUMN_STRING>,
			typename std::conditional<std::is_same<T, bool>::value, std::integral_constant<DataType, COLUMN_BOOL>,
			std::integral_constant<DataType, ERROR> >::type>::type>::type>::type>::type>::type>::type>::type retConst;
		return retConst::value;
	};
};
