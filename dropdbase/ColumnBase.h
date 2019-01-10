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
	ColumnBase(const std::string& name, int blockSize);
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
	const BlockBase<T>& AddBlock();

	/// <summary>
	/// Add new block with proper data into column
	/// </summary>
	/// <param name="data">Data to be inserted</param>
	/// <returns>Last block of column</returns>
	const BlockBase<T>& AddBlock(const std::vector<T>& data);

	/// <summary>
	/// Insert data into column considering empty space of last block and maximum size of blocks
	/// </summary>
	/// <param name="columnData">Data to be inserted</param>
	void InsertData(const std::vector<T>& columnData);

	/// <summary>
	/// Get all unique values for this column
	/// </summary>
	/// <returns>Array of unique values</returns>
	std::vector<T> GetUniqueBuckets() const;

	/// <summary>
	/// Insert null data into column
	/// </summary>
	/// <param name="length">Length of inserted data</param>
	void InsertNullData(int length);

	/// <summary>
	/// Returns type of ColumnBase
	/// </summary>
	/// <returns>Type of current column</returns>
	std::type_info GetColumnType() const
	{
		return dataType_;
	};
};


