
#pragma once
#include <string>
#include <typeinfo>
#include <vector>
#include <boost/log/trivial.hpp>

#include "BlockBase.h"
#include "ComplexPolygonFactory.h"
#include "PointFactory.h"
#include "IColumn.h"
#include "ConstraintViolationError.h"
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
    int64_t size_; // current number of not empty rows in a column, sumerized all blocks currentSize_
    int32_t blockSize_;
    std::map<int32_t, std::vector<std::unique_ptr<BlockBase<T>>>> blocks_;
    std::unordered_set<T> uniqueHashmap_;

    void setColumnStatistics();

    T min_ = std::numeric_limits<T>::lowest();
    T max_ = std::numeric_limits<T>::max();
    float avg_ = 0.0;
    T sum_ = T{};
    float initAvg_ = 0.0; // initial average is needed, because avg_ is constantly changing and we need unchable value for comparing in binary index
    bool initAvgIsSet_ = false;
    bool isNullable_;
    bool isUnique_;
    bool saveNecessary_;

public:
    ColumnBase(const std::string& name, int blockSize, bool isNullable = false, bool isUnique = false)
    : name_(name), size_(0), blockSize_(blockSize), blocks_(), isNullable_(false), isUnique_(false),
      saveNecessary_(true)
    {
        blocks_.emplace(-1, std::vector<std::unique_ptr<BlockBase<T>>>());
        SetIsNullable(isNullable);
        SetIsUnique(isUnique);
    }

    /// <summary>
    /// Try to insert new value into set, if it is possible (set does not contains this value)
    /// function returns true if it is not possible, function returns false
    /// </summary>
    /// <param name="value"> Value to be inserted </param>
    void InsertIntoHashmap(std::vector<T> values)
    {
        uniqueHashmap_.insert(values.begin(), values.end());
    }

    void InsertIntoHashmap(T value)
    {
        uniqueHashmap_.insert(value);
    }

    bool IsDuplicate(std::unordered_set<T>& temp_hashmap, T value)
    {
        return temp_hashmap.find(value) != temp_hashmap.end();
    }

    bool IsDuplicate(T value)
    {
        return uniqueHashmap_.find(value) != uniqueHashmap_.end();
    }

    std::unordered_set<T> GetHashmapCopy()
    {
        return uniqueHashmap_;
    }

    inline int32_t GetBlockSize() const
    {
        return blockSize_;
    };

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

    virtual std::pair<int64_t*, size_t> GetNullBitMaskForBlock(size_t blockIndex) override
    {
        auto block = GetBlocksList()[blockIndex];
        return std::make_pair(block->GetNullBitmask(), block->GetSize());
    }

    virtual bool GetIsNullable() const override
    {
        return isNullable_;
    }

    /// <summary>
    /// set isNullable_ flag, checking is there null value in column which doesnt allow to set FALSE flag
    /// </summary>
    /// <param name="isNullable">required isNullable_ value</param>
    virtual void SetIsNullable(bool isNullable) override
    {
        if (isNullable_ == isNullable)
        {
            // No change, do nothing
            return;
        }

        if (isNullable)
        {
            if (isUnique_)
            {
                throw constraint_violation_error(UNIQUE_CONSTRAINT_DROP_NOT_NULL,
                                                 "Could not drop NOT NULL constraint on column: " +
                                                     name_ + ", column has UNIQUE constraint (drop this constraint first)");
            }
            isNullable_ = true;
            BOOST_LOG_TRIVIAL(debug) << "Flag isNullable_ was set to TRUE for column named: " << name_ << ".";
        }
        else
        {
            bool isNullValue = false;
            for (auto const& mapBlock : blocks_)
            {
                for (int32_t i = 0; i < mapBlock.second.size() && !isNullValue; i++)
                {
                    int64_t* mask = mapBlock.second[i]->GetNullBitmask();

                    for (int32_t j = 0; j < mapBlock.second[i]->GetSize() && !isNullValue; j++)
                    {
                        const int8_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);

                        if (bit)
                        {
                            isNullValue = true;
                        }
                    }
                }
            }

            if (isNullValue)
            {
                throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
                                                 "Could not add NOT NULL constraint on column: " + name_ +
                                                     ", column contains null values");
            }
            else
            {
                isNullable_ = false;
                BOOST_LOG_TRIVIAL(debug)
                    << "Flag isNullable_ was set to FALSE for column named: " << name_ << ".";
            }
        }
    }

    virtual bool GetIsUnique() const override
    {
        return isUnique_;
    }

    /// <summary>
    /// set isUnique_ flag, checking is there is duplicity value or null value in column which dont allow to set TRUE flag
    /// </summary>
    /// <param name="isUnique">required isUnique_ value</param>
    virtual void SetIsUnique(bool isUnique) override
    {
        if (isUnique_ == isUnique)
        {
            // No change, do nothing
            return;
        }

        uniqueHashmap_.clear();

        if (isUnique)
        {
            if (isNullable_)
            {
                throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
                                                 "Could not add UNIQUE constraint on column: " + name_ +
                                                     ", column need to have NOT NULL constraint");
            }

            T duplicateData;
            bool duplicateFound = false;
            for (auto const& blocksId : blocks_)
            {
                for (int32_t i = 0; i < blocksId.second.size() && !duplicateFound; i++)
                {
                    auto data = blocksId.second[i]->GetData();

                    for (int32_t j = 0; j < blocksId.second[i]->GetSize() && !duplicateFound; j++)
                    {
                        if (!IsDuplicate(uniqueHashmap_, data[j]))
                        {
                            InsertIntoHashmap(data[j]);
                        }
                        else
                        {
                            duplicateFound = true;
                            duplicateData = data[j];
                        }
                    }
                }
            }

            if (!duplicateFound)
            {
                isUnique_ = true;
                BOOST_LOG_TRIVIAL(debug) << "Flag isUnique_ was set to TRUE for column named: " << name_ << ".";
            }
            else
            {
                throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                                                 "Could not add UNIQUE constraint on column: " + name_ +
                                                     ", column contains duplicate value: " +
                                                     std::to_string(duplicateData));
            }
        }

        else
        {
            isUnique_ = false;
            BOOST_LOG_TRIVIAL(debug) << "Flag isUnique_ was set to FALSE for column named: " << name_ << ".";
        }
    }

    virtual bool GetSaveNecessary() const override
    {
        return saveNecessary_;
    }

    virtual void SetSaveNecessaryToFalse() override
    {
        saveNecessary_ = false;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to FALSE for column named: " << name_ << ".";
    }

    virtual void SetSaveNecessaryToTrue() override
    {
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
    }

    virtual void SetColumnName(std::string newName) override
    {
        name_ = newName;
    }

    static std::vector<T> NullArray(int32_t length);

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
    const std::vector<BlockBase<T>*> GetBlocksList() const
    {
        std::vector<BlockBase<T>*> ret;

        for (auto& block : blocks_)
        {
            for (auto& ptr : block.second)
            {
                ret.emplace_back(ptr.get());
            }
        }

        return ret;
    };

    /// <summary>
    /// Resize column data blocks from source column to this column. If the source blocks are not
    /// fully filled, after resizing, the blocks will be fully filled, just the last block does not
    /// have to be fully filled. For this reason, the clustered indexing has to be dropped before
    /// this operation.
    /// </summary>
    /// <param name="srcColumnArg">The column whose data will be copied (resized).</param>
    virtual void ResizeColumn(IColumn* srcColumnArg) override
    {
        auto srcColumn = dynamic_cast<ColumnBase<T>*>(srcColumnArg);
        auto& srcBlocks = srcColumn->GetBlocksList();

        int32_t srcBlockIndex = 0;
        int32_t srcRowIndex = 0;
        int32_t dstBlockIndex = 0;
        int32_t dstRowIndex = 0;

        const int32_t newBlocksCount = (srcColumn->GetSize() + blockSize_ - 1) / blockSize_;

        // Add as many empty blocks as is needed for resizing
        for (int32_t i = 0; i < newBlocksCount; i++)
        {
            AddBlock();
        }

        // If a column is nullable, null bit mask values will be copied value by value, so it is much slower
        if (isNullable_)
        {
            while (srcBlockIndex < srcColumn->GetBlockCount())
            {
                const int64_t* nullBitMask = srcBlocks[srcBlockIndex]->GetNullBitmask();
                const bool isNullValue = NullValues::GetConcreteBitFromBitmask(nullBitMask, srcRowIndex);

                InsertDataOnSpecificPositionResizing(dstBlockIndex, dstRowIndex,
                                                     srcBlocks[srcBlockIndex]->GetData()[srcRowIndex],
                                                     -1, isNullValue);
                srcRowIndex++;
                dstRowIndex++;

                if (srcRowIndex == srcBlocks[srcBlockIndex]->GetSize())
                {
                    srcRowIndex = 0;
                    srcBlockIndex++;
                }

                if (dstRowIndex == blockSize_)
                {
                    dstRowIndex = 0;
                    dstBlockIndex++;
                }
            }
        }
        else
        {
            while (srcBlockIndex < srcColumn->GetBlockCount())
            {
                if (srcBlocks[srcBlockIndex]->GetSize() - srcRowIndex > static_cast<int64_t>(blockSize_ - dstRowIndex))
                {
                    // srcBlock[i].size_ > dstBlock
                    size_ += blockSize_ - dstRowIndex;
                    blocks_[-1][dstBlockIndex]->InsertDataInterval(srcBlocks[srcBlockIndex]->GetData(),
                                                                   srcRowIndex, blockSize_ - dstRowIndex);
                    srcRowIndex += blockSize_ - dstRowIndex;
                    dstBlockIndex += 1;
                    dstRowIndex = 0;
                }
                else
                {
                    if (srcBlocks[srcBlockIndex]->GetSize() - srcRowIndex ==
                        static_cast<int64_t>(blockSize_ - dstRowIndex))
                    {
                        // srcBlock[i].size_ == dstBlock
                        size_ += blockSize_ - dstRowIndex;
                        blocks_[-1][dstBlockIndex]->InsertDataInterval(srcBlocks[srcBlockIndex]->GetData(),
                                                                       srcRowIndex, blockSize_ - dstRowIndex);
                        srcBlockIndex += 1;
                        srcRowIndex = 0;
                        dstBlockIndex += 1;
                        dstRowIndex = 0;
                    }
                    else
                    {
                        // srcBlock[i].size_ < dstBlock
                        size_ += srcBlocks[srcBlockIndex]->GetSize() - srcRowIndex;
                        blocks_[-1][dstBlockIndex]->InsertDataInterval(srcBlocks[srcBlockIndex]->GetData(), srcRowIndex,
                                                                       srcBlocks[srcBlockIndex]->GetSize() - srcRowIndex);
                        dstRowIndex += srcBlocks[srcBlockIndex]->GetSize() - srcRowIndex;
                        srcBlockIndex += 1;
                        srcRowIndex = 0;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Add new empty block in column
    /// </summary>
    /// <returns>Last block of column</returns>
    BlockBase<T>& AddBlock(int32_t groupId = -1)
    {
        if (blocks_.find(groupId) == blocks_.end())
        {
            // key not found
            blocks_.emplace(groupId, std::vector<std::unique_ptr<BlockBase<T>>>());
        }

        blocks_[groupId].push_back(std::make_unique<BlockBase<T>>(*this));
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        return *(dynamic_cast<BlockBase<T>*>(blocks_[groupId].back().get()));
    }

    /// <summary>
    /// Add new block with proper data into column
    /// </summary>
    /// <param name="data">Data to be inserted</param>
    /// <returns>Last block of column</returns>
    BlockBase<T>& AddBlock(const std::vector<T>& data,
                           int32_t groupId = -1,
                           bool compress = false,
                           bool isCompressed = false,
                           bool countBlockStatistics = true)
    {
        blocks_[groupId].push_back(std::make_unique<BlockBase<T>>(data, *this, isCompressed,
                                                                  isNullable_, countBlockStatistics));
        auto& lastBlock = blocks_[groupId].back();
        if (lastBlock->IsFull() && !isCompressed && compress)
        {
            lastBlock->CompressData();
        }
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        size_ += data.size();
        return *(dynamic_cast<BlockBase<T>*>(blocks_[groupId].back().get()));
    }

    BlockBase<T>& AddBlock(std::unique_ptr<T[]>&& data,
                           int32_t dataSize,
                           int32_t allocationSize,
                           int32_t groupId = -1,
                           bool compress = false,
                           bool isCompressed = false,
                           bool countBlockStatistics = true)
    {
        blocks_[groupId].push_back(std::make_unique<BlockBase<T>>(std::move(data), dataSize,
                                                                  allocationSize, *this, isCompressed,
                                                                  isNullable_, countBlockStatistics));
        auto& lastBlock = blocks_[groupId].back();
        if (lastBlock->IsFull() && !isCompressed && compress)
        {
            lastBlock->CompressData();
        }
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        size_ += dataSize;
        return *(dynamic_cast<BlockBase<T>*>(blocks_[groupId].back().get()));
    }

    virtual size_t GetBlockSize(int32_t blockIndex) const override
    {
        return (GetBlocksList()[blockIndex])->GetSize();
    }

    virtual int64_t GetSize() const override
    {
        return size_;
    }

    /// <summary>
    /// Inserts data on proper position in column and split blocks, used with clustered indexes
    /// </summary>
    /// <param name="indexBlock">index of block where data will be inserted</param>
    /// <param name="indexInBlock">index in block where data will be inserted</param>
    /// <param name="columnData">data to insert<param>
    /// <param name="groupId">id of binary index group<param>
    /// <param name="isNullValue">whether data is null value flag<param>

    void InsertDataOnSpecificPosition(const int32_t indexBlock,
                                      const int32_t indexInBlock,
                                      const T& columnData,
                                      int32_t groupId = -1,
                                      bool isNullValue = false)
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

        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        // setColumnStatistics();
    }

    /// <summary>
    /// Inserts data on proper position in column without splitting blocks, used when resizing block size
    /// </summary>
    /// <param name="indexBlock">index of block where data will be inserted</param>
    /// <param name="indexInBlock">index in block where data will be inserted</param>
    /// <param name="columnData">data to insert<param>
    /// <param name="groupId">id of binary index group<param>
    /// <param name="isNullValue">whether data is null value flag<param>
    void InsertDataOnSpecificPositionResizing(int32_t indexBlock,
                                              int32_t indexInBlock,
                                              const T& columnData,
                                              int32_t groupId = -1,
                                              bool isNullValue = false)
    {
        BlockBase<T>& block = *(blocks_[groupId][indexBlock].get());

        if (block.IsFull())
        {
            BOOST_LOG_TRIVIAL(debug) << "The block with index "
                                     << indexBlock << " is full and data cannot be inserted into it when resizing block size for column named: "
                                     << name_ << ".";
        }

        size_ += 1;
        block.InsertDataOnSpecificPosition(indexInBlock, columnData, isNullValue);

        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
    }

    /// <summary>
    /// Splits block
    /// </summary>
    /// <param name="blockPtr">block that should be splitted</param>
    /// <param name="groupId">id of binary index group<param>
    void BlockSplit(std::unique_ptr<BlockBase<T>>& blockPtr, int32_t groupId = -1)
    {
        BlockBase<T>& block = *(blockPtr.get());
        std::vector<T> data1;
        std::vector<T> data2;
        const T* data = block.GetData();

        for (int32_t i = 0; i < block.GetSize(); i++)
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

        std::unique_ptr<BlockBase<T>> block1 =
            std::make_unique<BlockBase<T>>(data1, *this, block.IsCompressed(), block.IsNullable());
        std::unique_ptr<BlockBase<T>> block2 =
            std::make_unique<BlockBase<T>>(data2, *this, block.IsCompressed(), block.IsNullable());

        if (isNullable_)
        {
            int32_t bitMaskIdx = NullValues::GetBitMaskIdx((block.GetSize() - 1) / 2);
            int32_t shiftIdx = NullValues::GetShiftMaskIdx((block.GetSize() - 1) / 2);

            for (size_t i = 0; i < bitMaskIdx; i++)
            {
                block1->GetNullBitmask()[i] = block.GetNullBitmask()[i];
            }
            block1->GetNullBitmask()[bitMaskIdx] =
                NullValues::GetPartOfBitmaskByte(block.GetNullBitmask(), shiftIdx, bitMaskIdx);

            int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(block.BlockCapacity());

            for (size_t i = bitMaskIdx; i < bitMaskCapacity; i++)
            {
                int64_t tmp = (block.GetNullBitmask()[i] >> (shiftIdx + 1));
                if (bitMaskIdx + 1 < bitMaskCapacity)
                {
                    tmp |= NullValues::GetPartOfBitmaskByte(block.GetNullBitmask(), shiftIdx, bitMaskIdx + 1);
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
    void InsertData(const std::vector<T>& columnData, int32_t groupId = -1, bool compress = false)
    {
        int32_t startIdx = 0;

        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";

        if (blocks_[groupId].size() > 0 && !blocks_[groupId].back()->IsFull())
        {
            auto& lastBlock = blocks_[groupId].back();
            if (columnData.size() <= lastBlock->EmptyBlockSpace())
            {
                size_ += columnData.size();
                lastBlock->InsertData(columnData);
                if (compress && lastBlock->IsFull())
                {
                    lastBlock->CompressData();
                }
                // setColumnStatistics();
                return;
            }
            int32_t emptySpace = lastBlock->EmptyBlockSpace();
            size_ += emptySpace;
            lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
            if (compress && lastBlock->IsFull())
            {
                lastBlock->CompressData();
            }
            startIdx += emptySpace;
        }

        while (startIdx < columnData.size())
        {
            int32_t toCopy = columnData.size() - startIdx < blockSize_ ? columnData.size() - startIdx : blockSize_;
            AddBlock(std::vector<T>(columnData.cbegin() + startIdx, columnData.cbegin() + startIdx + toCopy),
                     groupId, compress, false);
            startIdx += toCopy;
        }
        // setColumnStatistics();
    }

    /// <summary>
    /// Insert data into column considering empty space of last block and maximum size of blocks
    /// </summary>
    /// <param name="columnData">Data to be inserted</param>
    void InsertData(const std::vector<T>& columnData,
                    const std::vector<int64_t>& nullMask,
                    int32_t groupId = -1,
                    bool compress = false)
    {
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        int32_t startIdx = 0;
        int32_t maskIdx = 0;
        if (blocks_[groupId].size() > 0 && !blocks_[groupId].back()->IsFull())
        {
            auto& lastBlock = blocks_[groupId].back();
            if (columnData.size() <= lastBlock->EmptyBlockSpace())
            {
                size_ += columnData.size();
                int32_t bitMaskStartIdx = lastBlock->GetSize();
                auto maskPtr = lastBlock->GetNullBitmask();
                for (int32_t i = bitMaskStartIdx; i < bitMaskStartIdx + columnData.size(); i++)
                {

                    if (NullValues::GetConcreteBitFromBitmask(nullMask.data(), maskIdx))
                    {
                        NullValues::SetBitInBitMask(maskPtr, i, 1);
                    }
                    maskIdx++;
                }
                lastBlock->InsertData(columnData);
                if (compress && lastBlock->IsFull())
                {
                    lastBlock->CompressData();
                }
                //setColumnStatistics();
                return;
            }

            int32_t emptySpace = lastBlock->EmptyBlockSpace();
            auto maskPtr = lastBlock->GetNullBitmask();
            size_ += emptySpace;
            int32_t bitMaskStartIdx = lastBlock->GetSize();
            lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
            for (int32_t i = bitMaskStartIdx; i < lastBlock->BlockCapacity(); i++)
            {
                if (NullValues::GetConcreteBitFromBitmask(nullMask.data(), maskIdx))
                {
                    NullValues::SetBitInBitMask(maskPtr, i, 1);
                }
                maskIdx++;
            }
            if (compress && lastBlock->IsFull())
            {
                lastBlock->CompressData();
            }
            startIdx += emptySpace;
        }

        while (startIdx < columnData.size())
        {
            int32_t toCopy = columnData.size() - startIdx < blockSize_ ? columnData.size() - startIdx : blockSize_;
            auto& block = AddBlock(std::vector<T>(columnData.cbegin() + startIdx,
                                                  columnData.cbegin() + startIdx + toCopy),
                                   groupId, compress, false);
            auto maskPtr = block.GetNullBitmask();
            for (int32_t i = 0; i < toCopy; i++)
            {
                if (NullValues::GetConcreteBitFromBitmask(nullMask.data(), maskIdx))
                {
                    NullValues::SetBitInBitMask(maskPtr, i, 1);
                }
                maskIdx++;
            }
            startIdx += toCopy;
        }
        // setColumnStatistics();
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
    void InsertNullData(int32_t length) override
    {
        std::vector<int64_t> nullMask(length, -1); // fill mask with bits 1
        InsertData(NullArray(length), nullMask);
    }

    void CopyDataToColumn(IColumn* destinationColumn)
    {
        auto toType = destinationColumn->GetColumnType();

        switch (toType)
        {
        case COLUMN_INT:
        {
            auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int32_t> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int32_t data = static_cast<int32_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        case COLUMN_LONG:
        {
            auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    int64_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int64_t> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int64_t data = static_cast<int64_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        case COLUMN_DOUBLE:
        {
            auto castedColumn = dynamic_cast<ColumnBase<double>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    int64_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<double> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        double data = static_cast<double>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        case COLUMN_FLOAT:
        {
            auto castedColumn = dynamic_cast<ColumnBase<float>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    int64_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<float> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        float data = static_cast<float>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        case COLUMN_INT8_T:
        {
            auto castedColumn = dynamic_cast<ColumnBase<int8_t>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    int64_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int8_t> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int8_t data = static_cast<int8_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        case COLUMN_STRING:
        {
            auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(destinationColumn);

            for (auto& block : blocks_)
            {
                int32_t blockCountOnIdx = block.second.size();
                for (int32_t i = 0; i < blockCountOnIdx; i++)
                {
                    int64_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<std::string> castedDataToCopy;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        std::string data = std::to_string(dataToCopy[j]);
                        castedDataToCopy.push_back(data);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                    auto nullBitMask = block.second.front()->GetNullBitmask();
                    newBlock.SetNullBitmask(
                        std::vector<int64_t>(nullBitMask, nullBitMask + NullValues::GetNullBitMaskSize(blockSize)));

                    block.second.erase(block.second.begin());
                }
            }
        }
        break;

        default:
            throw std::runtime_error("Attempt to execute unsupported column type conversion.");
            break;
        }
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

        for (auto& block : blocks_)
        {
            ret += block.second.size();
        }

        return ret;
    }

    virtual int64_t GetBlockSizeForIndex(int32_t blockIdx) const override
    {
        return GetBlocksList()[blockIdx]->GetSize();
    }
};

template <>
void ColumnBase<std::string>::SetIsUnique(bool isUnique);

template <>
void ColumnBase<ColmnarDB::Types::Point>::SetIsUnique(bool isUnique);

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::SetIsUnique(bool isUnique);

template <>
void ColumnBase<std::string>::CopyDataToColumn(IColumn* destinationColumn);

template <>
void ColumnBase<ColmnarDB::Types::Point>::CopyDataToColumn(IColumn* destinationColumn);

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::CopyDataToColumn(IColumn* destinationColumn);

template <>
std::vector<int32_t> ColumnBase<int32_t>::NullArray(int32_t length);

template <>
std::vector<float> ColumnBase<float>::NullArray(int32_t length);

template <>
std::vector<int64_t> ColumnBase<int64_t>::NullArray(int32_t length);

template <>
std::vector<double> ColumnBase<double>::NullArray(int32_t length);

template <>
std::vector<int8_t> ColumnBase<int8_t>::NullArray(int32_t length);

template <>
std::vector<std::string> ColumnBase<std::string>::NullArray(int32_t length);

template <>
std::vector<ColmnarDB::Types::Point> ColumnBase<ColmnarDB::Types::Point>::NullArray(int32_t length);

template <>
std::vector<ColmnarDB::Types::ComplexPolygon>
ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(int length);

template <>
void ColumnBase<int64_t>::setColumnStatistics();

template <>
void ColumnBase<float>::setColumnStatistics();

template <>
void ColumnBase<double>::setColumnStatistics();

template <>
void ColumnBase<ColmnarDB::Types::Point>::setColumnStatistics();

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::setColumnStatistics();

template <>
void ColumnBase<std::string>::setColumnStatistics();

template <>
void ColumnBase<int8_t>::setColumnStatistics();

template class ColumnBase<std::string>;
template class ColumnBase<ColmnarDB::Types::Point>;
template class ColumnBase<ColmnarDB::Types::ComplexPolygon>;