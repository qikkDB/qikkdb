#pragma once
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Compression/Compression.h"
#include "QueryEngine/GPUCore/NullValues.cuh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>

template <class T>
class ColumnBase;

/// <summary>
/// The main class representing block of data. Holds data of type T.
/// </summary>
/// <typeparam name="T">Data type of block.</typeparam>
template <class T>
class BlockBase
{
private:
    // statistical variable counting for each block - used by where evaluating
    T min_ = std::numeric_limits<T>::lowest();
    T max_ = std::numeric_limits<T>::max();
    float avg_ = 0.0;
    T sum_ = T{};

    int32_t groupId_ = -1; // index for group of blocks - binary index

    // these methods handle size_ and countOfNotNullValues_ of block
    // void setBlockStatistics(const std::vector<T>& data);
    void setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize);
    void updateBlockStatistics(const T& data, bool isNullValue);

    ColumnBase<T>& column_;
    size_t size_; // current number of not empty rows in a block, size_ is updated in BlockBase.cpp in updateBlockStatistics methods
    size_t countOfNotNullValues_; // current number of not null rows in a block, used and updated in BlockBase.cpp in setBlockStatistics and updateBlockStatistics methods
    size_t compressedSize_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;
    std::unique_ptr<nullmask_t[]> bitMask_;
    bool isCompressed_;
    bool isNullable_;   // flag indicating whether the block is able to contain null values
    bool wasRegistered_;
    bool isNullMaskRegistered_;
    bool saveNecessary_;    // flag indicating whether the block is modified (or new) and therefore it is necessary to persist

public:
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class filled with data.
    /// </summary>
    /// <param name="data">Data which will fill up the block.</param>
    /// <param name="column">Column that will hold this new block.</param>
    /// <param name="isCompressed">Flag indicating whether the block is compressed.</param>
    /// <param name="isNullable">Flag indicating whether the block is able to contain null values.</param>
    /// <param name="countBlockStatistics">Flag indicating whether statistics need to be ratified 
    /// (if false, use setBlockStatistics with concrete values, eg when loading from disk after persisting data even with already counted statistics).</param>
    /// <exception cref="std::length_error">Attempted to insert data larger than block size.</exception>
    BlockBase(const std::vector<T>& data,
              ColumnBase<T>& column,
              bool isCompressed = false,
              bool isNullable = false,
              bool countBlockStatistics = true)
    : column_(column), size_(0), countOfNotNullValues_(0), isCompressed_(isCompressed),
      isNullable_(isNullable), bitMask_(nullptr), wasRegistered_(false),
      isNullMaskRegistered_(false), saveNecessary_(true)
    {
        capacity_ = (isCompressed) ? data.size() : column.GetBlockSize();
        data_ = std::unique_ptr<T[]>(new T[capacity_]);

        if (column_.GetBlockSize() < data.size())
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }
        GPUMemory::hostPin(data_.get(), capacity_);
        wasRegistered_ = true;
        if (isNullable_)
        {
            int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
            bitMask_ = std::unique_ptr<nullmask_t[]>(new nullmask_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity * sizeof(nullmask_t));
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }
        std::copy(data.begin(), data.end(), data_.get());

        if (countBlockStatistics)
        {
            setBlockStatistics(data.size(), 0);
        }
    }
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class filled with data.
    /// </summary>
    /// <param name="data">Data which will fill up the block.</param>
    /// <param name="dataSize">Compressed data size.</param>
    /// <param name="allocationSize">Allocated size.</param>
    /// <param name="column">Column that will hold this new block.</param>
    /// <param name="isCompressed">Flag indicating whether the block is compressed.</param>
    /// <param name="isNullable">Flag indicating whether the block is able to contain null values.</param>
    /// <param name="countBlockStatistics">Flag indicating whether statistics need to be ratified
    /// (if false, use setBlockStatistics with concrete values, eg when loading from disk after persisting data even with already counted statistics).</param>
    /// <exception cref="std::length_error">Attempted to load data with size not equal to column size.</exception>
    /// <exception cref="std::length_error">Attempted to insert data larger than block size.</exception>
    BlockBase(std::unique_ptr<T[]>&& data,
              int32_t dataSize,
              int32_t allocationSize,
              ColumnBase<T>& column,
              bool isCompressed = false,
              bool isNullable = false,
              bool countBlockStatistics = true)
    : column_(column), size_(0), countOfNotNullValues_(0), isCompressed_(isCompressed),
      isNullable_(isNullable), bitMask_(nullptr), wasRegistered_(false),
      isNullMaskRegistered_(false), saveNecessary_(true)
    {
        if (allocationSize != column_.GetBlockSize())
        {
            throw std::length_error("Size of loaded data must be equal to block size");
        }

        capacity_ = (isCompressed) ? dataSize : column.GetBlockSize();
        data_ = std::move(data);

        if (column_.GetBlockSize() < dataSize)
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }


        GPUMemory::hostPin(data_.get(), capacity_);
        wasRegistered_ = true;
        if (isNullable_)
        {
            int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
            bitMask_ = std::unique_ptr<nullmask_t[]>(new nullmask_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity * sizeof(nullmask_t));
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }

        if (countBlockStatistics)
        {
            setBlockStatistics(dataSize, 0);
        }
    }
    
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class without data.
    /// </summary>
    /// <param name="column">Column that will hold this new empty block.</param>
    explicit BlockBase(ColumnBase<T>& column)
    : column_(column), size_(0), countOfNotNullValues_(0), capacity_(column_.GetBlockSize()),
      data_(new T[capacity_]), bitMask_(nullptr), isNullable_(column_.GetIsNullable()),
      wasRegistered_(false), isNullMaskRegistered_(false), saveNecessary_(true)
    {
        GPUMemory::hostPin(data_.get(), capacity_);
        wasRegistered_ = true;
        isCompressed_ = false;
        if (isNullable_)
        {
            int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
            bitMask_ = std::unique_ptr<nullmask_t[]>(new nullmask_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity * sizeof(nullmask_t));
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }
    }

    void setBlockStatistics(T min, T max, float avg, T sum, int32_t dataSize)
    {
        min_ = min;
        max_ = max;
        avg_ = avg;
        sum_ = sum;

        size_ += dataSize;
        saveNecessary_ = true;
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

    T* GetData()
    {
        return data_.get();
    }

    bool GetIsNullable()
    {
        return isNullable_;
    }

    /// <summary>
    /// Set isNullable_ flag with required value. If this value is TRUE, we need to allocate nullmasks for block. If FALSE, these nullmasks can be deleted.
    /// </summary>
    /// <param name="isNullable">Required isNullable_ value.</param>
    void SetIsNullable(const bool isNullable)
    {
        if (isNullable_ == isNullable)
        {
            // No change, do nothing
            return;
        }

        else if (isNullable)
        {
            const int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
            bitMask_ = std::unique_ptr<nullmask_t[]>(new nullmask_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity * sizeof(nullmask_t));
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
            isNullable_ = isNullable;
        }

        else
        {
            if (isNullMaskRegistered_)
            {
                GPUMemory::hostUnregister(bitMask_.get());
                isNullMaskRegistered_ = false;
            }
            bitMask_.reset();
            isNullable_ = isNullable;
        }
        saveNecessary_ = true;
    }

    nullmask_t* GetNullBitmask()
    {
        return bitMask_.get();
    }

    std::unique_ptr<nullmask_t[]> GetNullBitmaskPtr()
    {
        return std::move(bitMask_);
    }

    size_t GetSize() const
    {
        return size_;
    }

    size_t GetNullBitmaskSize() const
    {
        return NullValues::GetNullBitMaskSize(size_);
    }

    bool GetSaveNecessary() const
    {
        return saveNecessary_;
    }

    void SetSaveNecessaryToFalse()
    {
        saveNecessary_ = false;
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

    constexpr bool IsNullable() const
    {
        return isNullable_;
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
            throw std::length_error("BlockBase.h/InsertData(): Attempted to insert data larger "
                                    "than remaining block size");
        }
        std::copy(data.begin(), data.end(), data_.get() + size_);
        setBlockStatistics(data.size(), size_);
        saveNecessary_ = true;
    }

    /// <summary>
    /// Insert data into the current block.
    /// </summary>
    /// <param name="newData">Data, which contain part that is gonna be inserted into the block.</param>
    /// <param name="offset">Offset, which define beginning of inderted part of data.</param>
    /// <param name="length">Lenght of data, define by offset, that are gonna be inserted.</param>
    /// <exception cref="std::length_error">Attempted to insert data larger than remaining block size.</exception>
    void InsertDataInterval(const T* newData, size_t offset, size_t length)
    {
        if (EmptyBlockSpace() < length)
        {
            throw std::length_error("BlockBase.h/InsertDataInterval(): Attempted to insert data "
                                    "larger than remaining block size");
        }

        std::copy(newData + offset, newData + offset + length, data_.get() + size_);
        setBlockStatistics(length, size_);
        saveNecessary_ = true;
    }

    void SetNullBitmask(const std::vector<nullmask_t>& nullMask)
    {
        if (isNullable_)
        {
            std::copy(nullMask.begin(), nullMask.end(), bitMask_.get());

            // count statistics from whole block - setting nullmasks can change them
            setBlockStatistics(0, 0);
            saveNecessary_ = true;
        }
    }

    void SetNullBitmask(std::unique_ptr<nullmask_t[]>&& nullMask)
    {
        if (isNullable_)
        {
            if (bitMask_)
            {
                GPUMemory::hostUnregister(bitMask_.get());
                isNullMaskRegistered_ = false;
            }
            bitMask_ = std::move(nullMask);
            if (bitMask_)
            {
                int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
				GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
                isNullMaskRegistered_ = true;
            }
            // count statistics from whole block - setting nullmasks can change them
            setBlockStatistics(0, 0);
            saveNecessary_ = true;
        }
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
        if (isCompressed_) return;

        bool compressedSuccessfully = false;
        std::vector<T> dataCompressed;
        Compression::Compress(column_.GetColumnType(), data_.get(), size_, dataCompressed, min_,
                              max_, compressedSuccessfully);
        if (compressedSuccessfully)
        {
            GPUMemory::hostUnregister(data_.get());

            capacity_ = dataCompressed.size();
            size_ = dataCompressed.size();
            data_.reset();
            data_ = std::unique_ptr<T[]>(new T[capacity_]);

            GPUMemory::hostPin(data_.get(), capacity_);
            std::copy(dataCompressed.begin(), dataCompressed.end(), data_.get());

            isCompressed_ = true;
            saveNecessary_ = true;
        }
    }

    void DecompressData()
    {
        if (!isCompressed_) return;

        bool decompressedSuccessfully = false;
        std::vector<T> dataDecompressed;
        int64_t uncompressedElementsCount = Compression::GetUncompressedDataElementsCount(data_.get());
        int64_t compressionBlocksCount = Compression::GetCompressionBlocksCount(data_.get());
        Compression::Decompress(column_.GetColumnType(), data_.get(), size_, dataDecompressed,
                                uncompressedElementsCount, compressionBlocksCount, min_, max_,
                                decompressedSuccessfully);
        if (decompressedSuccessfully)
        {
            GPUMemory::hostUnregister(data_.get());

            capacity_ = column_.GetBlockSize();
            size_ = column_.GetBlockSize();
            data_.reset();
            data_ = std::unique_ptr<T[]>(new T[capacity_]);

            GPUMemory::hostPin(data_.get(), capacity_);
            std::copy(dataDecompressed.begin(), dataDecompressed.end(), data_.get());

            isCompressed_ = false;
            saveNecessary_ = true;
        }
    }

    /// <summary>
    /// Inserts data on proper position in block
    /// </summary>
    /// <param name="index">index in block where data will be inserted</param>
    /// <param name="data">value to insert<param>
    /// <param name="isNullValue">whether data is null value flag<param>
    /// <exception cref="std::length_error">Attempted to insert data larger than remaining block size.</exception>
    void InsertDataOnSpecificPosition(int32_t index, const T& data, nullmask_t isNullValue = false)
    {
        if (EmptyBlockSpace() == 0)
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }

        else if (index < size_)
        {
            std::move_backward(data_.get() + index, data_.get() + size_, data_.get() + size_ + 1);

            int32_t bitMaskIdx = NullValues::GetBitMaskIdx(index);
            int32_t shiftIdx = NullValues::GetShiftMaskIdx(index);

            nullmask_t last = isNullValue ? static_cast<nullmask_t>(1U) : static_cast<nullmask_t>(0U);

            if (isNullable_)
            {
                for (size_t i = shiftIdx; i < (sizeof(nullmask_t) * 8); i++)
                {
                    nullmask_t tmp = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), bitMaskIdx, i);

                    if (last != tmp)
                    {
                        NullValues::SetBitInBitMask(bitMask_.get(), bitMaskIdx, i, last);
                        last = tmp;
                    }
                }

                bitMaskIdx++;
                int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(capacity_);
                for (size_t i = bitMaskIdx; i < bitMaskCapacity; i++)
                {
                    nullmask_t tmp = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), i,
                                                                       (sizeof(nullmask_t) * 8 - 1));
                    bitMask_[i] <<= static_cast<nullmask_t>(1U);
                    NullValues::SetBitInBitMask(bitMask_.get(), i, 0, last);
                    last = tmp;
                }
            }
        }
        else if (isNullValue)
        {
            NullValues::SetBitInBitMask(bitMask_.get(), index, isNullValue);
        }
        data_[index] = data;
        updateBlockStatistics(data, isNullValue);
        saveNecessary_ = true;
    }

    ~BlockBase()
    {

        if (wasRegistered_)
        {
            GPUMemory::hostUnregister(data_.get());
        }
        if (isNullMaskRegistered_)
        {
            GPUMemory::hostUnregister(bitMask_.get());
        }
    }

    BlockBase(const BlockBase&) = delete;
    BlockBase& operator=(const BlockBase&) = delete;
};
