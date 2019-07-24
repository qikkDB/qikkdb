
#pragma once
#include <string>
#include <typeinfo>
#include <vector>

#include "BlockBase.h"
#include "ComplexPolygonFactory.h"
#include "IColumn.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"

namespace std
{
template <>
struct hash<ColmnarDB::Types::Point>
{
    size_t operator()(const ColmnarDB::Types::Point& x) const
    {
        static_assert(sizeof(size_t) == 8, "size_t is not 8 bytes");
        float latitude = x.geopoint().latitude();
        float longitude = x.geopoint().longitude();
        int32_t* iLatitude = reinterpret_cast<int32_t*>(&latitude);
        int32_t* iLongitude = reinterpret_cast<int32_t*>(&longitude);
        return static_cast<size_t>(*iLatitude) | (static_cast<size_t>(*iLongitude) << 32);
    }
};

template <>
struct hash<ColmnarDB::Types::ComplexPolygon>
{
    size_t operator()(const ColmnarDB::Types::ComplexPolygon& x) const
    {
        std::string wkt = ComplexPolygonFactory::WktFromPolygon(x);
        return std::hash<std::string>{}(wkt);
    }
};


template <>
struct equal_to<ColmnarDB::Types::Point>
{
    bool operator()(const ColmnarDB::Types::Point& lhs, const ColmnarDB::Types::Point& rhs) const
    {
        if (std::abs(lhs.geopoint().latitude() - rhs.geopoint().latitude()) >= 0.0001f ||
            std::abs(lhs.geopoint().longitude() - rhs.geopoint().longitude()) >= 0.0001f)
        {
            return false;
        }
        return true;
    }
};

template <>
struct equal_to<ColmnarDB::Types::ComplexPolygon>
{
    bool operator()(const ColmnarDB::Types::ComplexPolygon& lhs, const ColmnarDB::Types::ComplexPolygon& rhs) const
    {
        if (lhs.polygons_size() != rhs.polygons_size())
        {
            return false;
        }

        int32_t polySize = lhs.polygons_size();
        for (int32_t i = 0; i < polySize; i++)
        {
            if (lhs.polygons(i).geopoints_size() != rhs.polygons(i).geopoints_size())
            {
                return false;
            }
            int32_t pointSize = lhs.polygons(i).geopoints_size();
            for (int32_t j = 0; j < pointSize; j++)
            {

                if (std::abs(lhs.polygons(i).geopoints(j).latitude() -
                             rhs.polygons(i).geopoints(j).latitude()) >= 0.0001f ||
                    std::abs(lhs.polygons(i).geopoints(j).longitude() -
                             rhs.polygons(i).geopoints(j).longitude()) >= 0.0001f)
                {
                    return false;
                }
            }
        }
        return true;
    }
};
} // namespace std

template <class T>
class ColumnBase : public IColumn
{
private:
	std::string name_;
	int64_t size_;
	int blockSize_;
	std::map<int32_t, std::vector<std::unique_ptr<BlockBase<T>>>> blocks_;

    void setColumnStatistics();

	T min_ = std::numeric_limits<T>::lowest();
	T max_ = std::numeric_limits<T>::max();
	float avg_ = 0.0;
	T sum_ = T{};
	float initAvg_ = 0.0; //initial average is needed, because avg_ is constantly changing and we need unchable value for comparing in binary index
	bool initAvgIsSet_ = false;
	bool isNullable_;

public:
	ColumnBase(const std::string& name, int blockSize, bool isNullable = false) :
		name_(name), size_(0), blockSize_(blockSize), blocks_(), isNullable_(isNullable)
	{
		blocks_.emplace(-1, std::vector<std::unique_ptr<BlockBase<T>>>());
	}

	inline int GetBlockSize() const { return blockSize_; };

	virtual const std::string& GetName() const override
	{
		return name_;
	}

	virtual float GetInitAvg() const override
	{
		return initAvg_;
	}

	virtual bool GetInitAvgIsSet() const override
	{
		return initAvgIsSet_;
	}

	virtual std::pair<int8_t*, size_t> GetNullBitMaskForBlock(size_t blockIndex) override
	{
		auto block = GetBlocksList()[blockIndex];
		return std::make_pair(block->GetNullBitmask(), block->GetSize());
	}

	virtual bool GetIsNullable() const override
	{
		return isNullable_;
	}

	virtual void SetIsNullable(bool isNullable) override
	{
		isNullable_ = isNullable;
	}

	static std::vector<T> NullArray(int length);

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

	/// <summary>
	/// Blocks getter
	/// </summary>
	/// <returns>List of blocks in current column</returns>
	const std::vector<BlockBase<T> *> GetBlocksList() const
	{
		std::vector<BlockBase<T> *> ret;

		for (auto& stuff : blocks_)
		{
			for (auto& ptr : stuff.second)
			{
				ret.emplace_back(ptr.get());
			}
		}

		return ret;
	};

	/// <summary>
	/// Add new block in column
	/// </summary>
	/// <returns>Last block of column</returns>
	BlockBase<T>& AddBlock(int groupId = -1)
	{
		if (blocks_.find(groupId) == blocks_.end())
		{
			// key not found
			blocks_.emplace(groupId, std::vector<std::unique_ptr<BlockBase<T>>>());
		}

		blocks_[groupId].push_back(std::make_unique<BlockBase<T>>(*this));
		return *(dynamic_cast<BlockBase<T>*>(blocks_[groupId].back().get()));
	}

	/// <summary>
	/// Add new block with proper data into column
	/// </summary>
	/// <param name="data">Data to be inserted</param>
	/// <returns>Last block of column</returns>
	BlockBase<T>& AddBlock(const std::vector<T>& data, int groupId = -1, bool compress = false, bool isCompressed = false)
	{
		blocks_[groupId].push_back(std::make_unique<BlockBase<T>>(data, *this, isCompressed, isNullable_));
		auto & lastBlock = blocks_[groupId].back();
		if (lastBlock->IsFull() && !isCompressed && compress)
		{
			lastBlock->CompressData();
		}
		return *(dynamic_cast<BlockBase<T>*>(blocks_[groupId].back().get()));
	}

    std::tuple<int, int, int>
    FindIndexAndRange(int indexBlock, int indexInBlock, int range, const T& columnData, int groupId = -1, bool isNullValue = false)
    {
        int remainingRange = range;
        int blockRange = 0;
        int newRange = 0;
        int newIndexInBlock = indexInBlock;
        int newIndexBlock = indexBlock;
        int startIndexInCurrentBlock = indexInBlock;
        bool reachEnd = true;
        bool found = false;
		T nextBlockMin;
		T currentBlockMin;
		T currentMax;

		if (blocks_[groupId].size() == 0)
        {
            BlockBase<T>& block = AddBlock();
            newIndexBlock = 0;
            std::tie(newIndexInBlock, newRange, reachEnd) =
                block.FindIndexAndRange(indexInBlock, range, columnData, isNullValue);
		}
		else if (blocks_[groupId].size() == 1)
        {
            BlockBase<T>& block = *(blocks_[groupId][0].get());
            newIndexBlock = 0;
            std::tie(newIndexInBlock, newRange, reachEnd) =
                block.FindIndexAndRange(indexInBlock, range, columnData, isNullValue);
        }
        else
        {
			if (isNullValue)
			{

				while ((reachEnd && indexBlock < blocks_[groupId].size()) && remainingRange > 0) 
				{
					BlockBase<T>& block = *(blocks_[groupId][indexBlock].get());

					int tempIndexInBlock;
					std::tie(tempIndexInBlock, blockRange, reachEnd) =
						block.FindIndexAndRange(startIndexInCurrentBlock, remainingRange, columnData, isNullValue);
					newRange += blockRange;

					startIndexInCurrentBlock = 0;
					remainingRange -= block.GetSize() - startIndexInCurrentBlock;

					indexBlock++;
				}
			}
			else 
			{
				BlockBase<T> *block = (blocks_[groupId][indexBlock].get());

				if (isNullable_)
				{
					while (block->GetIsFullOfNullValue())
					{
						remainingRange -= block->GetSize();
						startIndexInCurrentBlock = 0;

						indexBlock++;
						block = (blocks_[groupId][indexBlock].get());
					}

					int bitMaskIdx = (startIndexInCurrentBlock / (sizeof(char) * 8));
					int shiftIdx = (startIndexInCurrentBlock % (sizeof(char) * 8));
					int nullValueCount = 0;

					while (((block->GetNullBitmask()[bitMaskIdx] >> shiftIdx) & 1) == 1)
					{
						startIndexInCurrentBlock++;
						nullValueCount++;
						bitMaskIdx = (startIndexInCurrentBlock / (sizeof(char) * 8));
						shiftIdx = (startIndexInCurrentBlock % (sizeof(char) * 8));
					}

					remainingRange -= nullValueCount;
				}

				if (indexBlock == blocks_[groupId].size() - 1)
				{
					BlockBase<T>& block = *(blocks_[groupId][indexBlock].get());

					std::tie(newIndexInBlock, newRange, reachEnd) =
						block.FindIndexAndRange(startIndexInCurrentBlock, remainingRange, columnData, isNullValue);
					newIndexBlock = indexBlock;
				}
				
				else
				{
					for (int i = indexBlock; i < blocks_[groupId].size() && reachEnd && remainingRange > 0; i++)
					{
						BlockBase<T>& block = *(blocks_[groupId][i].get());

						currentBlockMin = block.GetData()[startIndexInCurrentBlock];

						if ((i + 1) != blocks_[groupId].size())
						{
							BlockBase<T>& nextBlock = *(blocks_[groupId][i + 1].get());
							nextBlockMin = nextBlock.GetData()[0];
						}

						if (remainingRange >= block.GetSize() - startIndexInCurrentBlock)
						{
							currentMax = block.GetData()[block.GetSize() - 1];
						}
						else
						{
							currentMax = block.GetData()[startIndexInCurrentBlock + remainingRange];
						}

						if (columnData >= currentBlockMin &&
							(remainingRange <= block.GetSize() - startIndexInCurrentBlock ||
							(columnData <= currentMax || (i == blocks_[groupId].size() - 1 || columnData <= nextBlockMin))))
						{
							int tempIndexInBlock;
							std::tie(tempIndexInBlock, blockRange, reachEnd) =
								block.FindIndexAndRange(startIndexInCurrentBlock, remainingRange, columnData, isNullValue);
							if (!found)
							{
								newIndexInBlock = tempIndexInBlock;
								newIndexBlock = i;
								found = true;
							}
							newRange += blockRange;
						}

						remainingRange -= block.GetSize() - startIndexInCurrentBlock;
						startIndexInCurrentBlock = 0;
					}
				}
			}
        }
        return std::make_tuple(newIndexBlock, newIndexInBlock, newRange);
    }

	virtual int64_t GetSize() const override
	{
		return size_;
	}

    void InsertDataOnSpecificPosition(int indexBlock, int indexInBlock, const T& columnData, int groupId = -1, bool isNullValue = false)
    {
		size_ += 1;

        if (blocks_[groupId].size() == 0)
        {
            AddBlock();
        }
        BlockBase<T>& block = *(blocks_[groupId][indexBlock].get());
        block.InsertDataOnSpecificPosition(indexInBlock, columnData, isNullValue);

        if (block.IsFull())
        {
            BlockSplit(blocks_[groupId][indexBlock]);
        }

        setColumnStatistics();
    }

    void BlockSplit(std::unique_ptr<BlockBase<T>>& blockPtr, int groupId = -1)
    {
        BlockBase<T>& block = *(blockPtr.get());
        std::vector<T> data1;
        std::vector<T> data2;
        const T* data = block.GetData();

        for (int i = 0; i < block.GetSize(); i++)
        {
            if (i < block.GetSize() / 2)
            {
                data1.push_back(data[i]);
            }
            else
            {
                data2.push_back(data[i]);
            }
        }

        std::unique_ptr<BlockBase<T>> block1 = std::make_unique<BlockBase<T>>(data1, *this, block.IsCompressed(), block.IsNullable());
        std::unique_ptr<BlockBase<T>> block2 = std::make_unique<BlockBase<T>>(data2, *this, block.IsCompressed(), block.IsNullable());

		if (isNullable_)
		{
			int bitMaskIdx = (((block.GetSize() - 1) / 2) / (sizeof(char) * 8));
			int shiftIdx = (((block.GetSize() - 1) / 2) % (sizeof(char) * 8));

			for (size_t i = 0; i < bitMaskIdx; i++)
			{
				block1->GetNullBitmask()[i] = block.GetNullBitmask()[i];
			}
			block1->GetNullBitmask()[bitMaskIdx] = ((1 << (shiftIdx + 1)) - 1) & block.GetNullBitmask()[bitMaskIdx];


			int32_t bitMaskCapacity = ((block.BlockCapacity() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));

			for (size_t i = bitMaskIdx; i < bitMaskCapacity; i++)
			{
				int8_t tmp = block.GetNullBitmask()[i] >> (shiftIdx + 1);

				if (bitMaskIdx + 1 < bitMaskCapacity)
				{
					tmp |= ((1 << (shiftIdx + 1)) - 1) & block.GetNullBitmask()[bitMaskIdx + 1];
				}
				block2->GetNullBitmask()[i - bitMaskIdx] = tmp;
			}
		}
        auto blockIndex = std::find(blocks_[groupId].begin(), blocks_[groupId].end(), blockPtr);
		int32_t blockIdx = blockIndex - blocks_[groupId].begin();
        blocks_[groupId].erase(blockIndex);

        blocks_[groupId].insert(blocks_[groupId].begin() + blockIdx, std::move(block2));
        blocks_[groupId].insert(blocks_[groupId].begin() + blockIdx, std::move(block1));
    }

    /// <summary>
    /// Insert data into column considering empty space of last block and maximum size of blocks
    /// </summary>
    /// <param name="columnData">Data to be inserted</param>
	void InsertData(const std::vector<T>& columnData, int groupId = -1, bool compress = false)
	{
		size_ += columnData.size();
		int startIdx = 0;
		if (blocks_[groupId].size() > 0 && !blocks_[groupId].back()->IsFull())
		{
			auto & lastBlock = blocks_[groupId].back();
			if (columnData.size() <= lastBlock->EmptyBlockSpace())
			{
				lastBlock->InsertData(columnData);
				if (compress && lastBlock->IsFull())
				{
					lastBlock->CompressData();
				}
				setColumnStatistics();
				return;
			}
			int emptySpace = lastBlock->EmptyBlockSpace();
			lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
			if (compress && lastBlock->IsFull())
			{
				lastBlock->CompressData();
			}
			startIdx += emptySpace;
		}

		while (startIdx < columnData.size())
		{
			int toCopy = columnData.size() - startIdx < blockSize_
				? columnData.size() - startIdx
				: blockSize_;
			AddBlock(std::vector<T>(columnData.cbegin() + startIdx, columnData.cbegin() + startIdx + toCopy), groupId, compress, false);
			startIdx += toCopy;
		}
		setColumnStatistics();
	}

	/// <summary>
    /// Insert data into column considering empty space of last block and maximum size of blocks
    /// </summary>
    /// <param name="columnData">Data to be inserted</param>
	void InsertData(const std::vector<T>& columnData, const std::vector<int8_t>& nullMask, int groupId = -1, bool compress = false)
	{
		size_ += columnData.size();
		int startIdx = 0;
		int maskIdx = 0;
		if (blocks_[groupId].size() > 0 && !blocks_[groupId].back()->IsFull())
		{
			auto & lastBlock = blocks_[groupId].back();
			if (columnData.size() <= lastBlock->EmptyBlockSpace())
			{
				lastBlock->InsertData(columnData);
				auto maskPtr = lastBlock->GetNullBitmask();
				int bitMaskStartIdx = lastBlock->BlockCapacity() - lastBlock->EmptyBlockSpace() - 1;
				for(int i = bitMaskStartIdx; i < bitMaskStartIdx+columnData.size(); i++)
				{
					int nullMaskOffset = maskIdx / (sizeof(char) * 8);
					int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
					maskIdx++;
					if((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
					{
						int bitMaskIdx = (i / (sizeof(char)*8));
						maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char)*8));
					}
				}
				if (compress && lastBlock->IsFull())
				{
					lastBlock->CompressData();
				}
				setColumnStatistics();
				return;
			}
			int emptySpace = lastBlock->EmptyBlockSpace();
			auto maskPtr = lastBlock->GetNullBitmask();
			int bitMaskStartIdx = lastBlock->BlockCapacity() - lastBlock->EmptyBlockSpace() - 1;
			lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
			for(int i = bitMaskStartIdx; i < lastBlock->BlockCapacity(); i++)
			{
				int nullMaskOffset = maskIdx / (sizeof(char) * 8);
				int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
				maskIdx++;
				if ((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
				{
					int bitMaskIdx = (i / (sizeof(char)*8));
					maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char)*8));
				}
			}
			if (compress && lastBlock->IsFull())
			{
				lastBlock->CompressData();
			}
			startIdx += emptySpace;
		}

		while (startIdx < columnData.size())
		{
			int toCopy = columnData.size() - startIdx < blockSize_
				? columnData.size() - startIdx
				: blockSize_;
			auto& block = AddBlock(std::vector<T>(columnData.cbegin() + startIdx, columnData.cbegin() + startIdx + toCopy), groupId, compress, false);
			auto maskPtr = block.GetNullBitmask();
			for(int i = 0; i < toCopy; i++)
			{
				int nullMaskOffset = maskIdx / (sizeof(char) * 8);
				int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
				maskIdx++;
				if ((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
				{
					int bitMaskIdx = (i / (sizeof(char)*8));
					maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char)*8));
				}
			}
			startIdx += toCopy;
		}
		setColumnStatistics();
	}

    /// <summary>
    /// Get all unique values for this column
    /// </summary>
    /// <returns>Array of unique values</returns>
    std::vector<T> GetUniqueBuckets() const
    {
        std::unordered_set<T> dataSet;
        auto& floatBlocks = GetBlocksList();
        for (const auto& block : floatBlocks)
        {
            for (size_t i = 0; i < block->GetSize(); i++)
            {
                dataSet.insert(block->GetData()[i]);
            }
        }
        return std::vector<T>(dataSet.cbegin(), dataSet.cend());
    }

    /// <summary>
    /// Insert null data into column
    /// </summary>
    /// <param name="length">Length of inserted data</param>
    void InsertNullData(int length) override
    {
		std::vector<int8_t> nullMask(length, -1);	// fill mask with bits 1
        InsertData(NullArray(length), nullMask);
    }

    /// <summary>
    /// Returns type of ColumnBase
    /// </summary>
    /// <returns>Type of current column</returns>
    virtual DataType GetColumnType() const override
    {
		return ::GetColumnType<T>();
    };

    virtual int32_t GetBlockCount() const override
    {
		int32_t ret = 0;

		//TODO preiterovat celu mapu a zosumovat bloky
		for (auto& stuff : blocks_)
		{
			ret += stuff.second.size();
		}

		return ret;
	}
};
