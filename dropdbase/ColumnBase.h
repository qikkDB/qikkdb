
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
    int64_t size_;
    int blockSize_;
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
    : name_(name), size_(0), blockSize_(blockSize), blocks_(), saveNecessary_(true)
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

    inline int GetBlockSize() const
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

    virtual std::pair<int8_t*, size_t> GetNullBitMaskForBlock(size_t blockIndex) override
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
        if (!isNullable)
        {
            bool isNullValue = false;
            for (auto const& mapBlock : blocks_)
            {
                for (int32_t i = 0; i < mapBlock.second.size() && !isNullValue; i++)
                {
                    auto data = mapBlock.second[i]->GetData();
                    int8_t* mask = mapBlock.second[i]->GetNullBitmask();

                    for (int32_t j = 0; j < mapBlock.second[i]->GetSize() && !isNullValue; j++)
                    {
                        int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;

                        if (bit)
                        {
                            isNullValue = true;
                        }
                    }
                }
            }

            if (!isNullValue)
            {
                isNullable_ = false;
                BOOST_LOG_TRIVIAL(debug)
                    << "Flag isNullable_ was set to FALSE for column named: " << name_ << ".";
            }
            else
            {
                throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
                                                 "Could not add NOT NULL constraint on column: " + name_ +
                                                     ", column contains null values");
            }
        }

        else
        {
            isNullable_ = true;
            BOOST_LOG_TRIVIAL(debug) << "Flag isNullable_ was set to TRUE for column named: " << name_ << ".";
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
                    int8_t* mask = blocksId.second[i]->GetNullBitmask();

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

    virtual void SetColumnName(std::string newName) override
    {
        name_ = newName;
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
                           int groupId = -1,
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
                           int groupId = -1,
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
    /// Inserts data on proper position in column
    /// </summary>
    /// <param name="indexBlock">index of block where data will be inserted</param>
    /// <param name="indexInBlock">index in block where data will be inserted</param>
    /// <param name="columnData">data to insert<param>
    /// <param name="groupId">id of binary index group<param>
    /// <param name="isNullValue">whether data is null value flag<param>
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

        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        // setColumnStatistics();
    }

    /// <summary>
    /// Splits block
    /// </summary>
    /// <param name="blockPtr">block that should be splitted</param>
    /// <param name="groupId">id of binary index group<param>
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

        std::unique_ptr<BlockBase<T>> block1 =
            std::make_unique<BlockBase<T>>(data1, *this, block.IsCompressed(), block.IsNullable());
        std::unique_ptr<BlockBase<T>> block2 =
            std::make_unique<BlockBase<T>>(data2, *this, block.IsCompressed(), block.IsNullable());

        if (isNullable_)
        {
            int bitMaskIdx = (((block.GetSize() - 1) / 2) / (sizeof(char) * 8));
            int shiftIdx = (((block.GetSize() - 1) / 2) % (sizeof(char) * 8));

            for (size_t i = 0; i < bitMaskIdx; i++)
            {
                block1->GetNullBitmask()[i] = block.GetNullBitmask()[i];
            }
            block1->GetNullBitmask()[bitMaskIdx] =
                ((1 << (shiftIdx + 1)) - 1) & block.GetNullBitmask()[bitMaskIdx];


            int32_t bitMaskCapacity =
                ((block.BlockCapacity() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));

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
        int startIdx = 0;

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
            int emptySpace = lastBlock->EmptyBlockSpace();
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
            int toCopy = columnData.size() - startIdx < blockSize_ ? columnData.size() - startIdx : blockSize_;
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
                    const std::vector<int8_t>& nullMask,
                    int groupId = -1,
                    bool compress = false)
    {
        saveNecessary_ = true;
        BOOST_LOG_TRIVIAL(debug) << "Flag saveNecessary_ was set to TRUE for column named: " << name_ << ".";
        int startIdx = 0;
        int maskIdx = 0;
        if (blocks_[groupId].size() > 0 && !blocks_[groupId].back()->IsFull())
        {
            auto& lastBlock = blocks_[groupId].back();
            if (columnData.size() <= lastBlock->EmptyBlockSpace())
            {
                size_ += columnData.size();
                int bitMaskStartIdx = lastBlock->GetSize();
                lastBlock->InsertData(columnData);
                auto maskPtr = lastBlock->GetNullBitmask();
                for (int i = bitMaskStartIdx; i < bitMaskStartIdx + columnData.size(); i++)
                {
                    int nullMaskOffset = maskIdx / (sizeof(char) * 8);
                    int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
                    maskIdx++;
                    if ((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
                    {
                        int bitMaskIdx = (i / (sizeof(char) * 8));
                        maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char) * 8));
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
            size_ += emptySpace;
            int bitMaskStartIdx = lastBlock->GetSize();
            lastBlock->InsertData(std::vector<T>(columnData.cbegin(), columnData.cbegin() + emptySpace));
            for (int i = bitMaskStartIdx; i < lastBlock->BlockCapacity(); i++)
            {
                int nullMaskOffset = maskIdx / (sizeof(char) * 8);
                int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
                maskIdx++;
                if ((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
                {
                    int bitMaskIdx = (i / (sizeof(char) * 8));
                    maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char) * 8));
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
            int toCopy = columnData.size() - startIdx < blockSize_ ? columnData.size() - startIdx : blockSize_;
            auto& block = AddBlock(std::vector<T>(columnData.cbegin() + startIdx,
                                                  columnData.cbegin() + startIdx + toCopy),
                                   groupId, compress, false);
            auto maskPtr = block.GetNullBitmask();
            for (int i = 0; i < toCopy; i++)
            {
                int nullMaskOffset = maskIdx / (sizeof(char) * 8);
                int nullMaskShiftOffset = maskIdx % (sizeof(char) * 8);
                maskIdx++;
                if ((nullMask[nullMaskOffset] >> nullMaskShiftOffset) & 1)
                {
                    int bitMaskIdx = (i / (sizeof(char) * 8));
                    maskPtr[bitMaskIdx] |= 1 << (i % (sizeof(char) * 8));
                }
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
    void InsertNullData(int length) override
    {
        std::vector<int8_t> nullMask(length, -1); // fill mask with bits 1
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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int32_t> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int32_t data = static_cast<int32_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int64_t> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int64_t data = static_cast<int64_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<double> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        double data = static_cast<double>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<float> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        float data = static_cast<float>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<int8_t> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        int8_t data = static_cast<int8_t>(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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
                    int8_t* mask = block.second.front()->GetNullBitmask();

                    auto dataToCopy = block.second.front()->GetData();
                    auto blockSize = block.second.front()->GetSize();
                    std::vector<std::string> castedDataToCopy;
                    std::vector<int8_t> newNullMask;

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        std::string data = std::to_string(dataToCopy[j]);
                        castedDataToCopy.push_back(data);

                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = (mask[bitMaskIdx] >> shiftIdx) & 1;
                        newNullMask.push_back(bit);
                    }

                    auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);

                    for (int32_t j = 0; j < blockSize; j++)
                    {
                        const int32_t bitMaskIdx = (j / (sizeof(int8_t) * 8));
                        const int32_t shiftIdx = (j % (sizeof(int8_t) * 8));
                        const int8_t bit = newNullMask[j] << shiftIdx;
                        newBlock.GetNullBitmask()[bitMaskIdx] |= bit;
                    }

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

        // TODO preiterovat celu mapu a zosumovat bloky
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
std::vector<int32_t> ColumnBase<int32_t>::NullArray(int length);

template <>
std::vector<float> ColumnBase<float>::NullArray(int length);

template <>
std::vector<int64_t> ColumnBase<int64_t>::NullArray(int length);

template <>
std::vector<double> ColumnBase<double>::NullArray(int length);

template <>
std::vector<int8_t> ColumnBase<int8_t>::NullArray(int length);

template <>
std::vector<std::string> ColumnBase<std::string>::NullArray(int length);

template <>
std::vector<ColmnarDB::Types::Point> ColumnBase<ColmnarDB::Types::Point>::NullArray(int length);

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