#pragma once
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Compression/Compression.h"

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
    T min_ = std::numeric_limits<T>::lowest();
    T max_ = std::numeric_limits<T>::max();
    float avg_ = 0.0;
    T sum_ = T{};
    bool isFullOfNullValue_ = false;
    int32_t groupId_ = -1; // index for group of blocks - binary index

    // these methods handle size_ of block
    void setBlockStatistics(const std::vector<T>& data);
    void setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize);
    void updateBlockStatistics(const T& data, bool isNullValue);

    ColumnBase<T>& column_;
    size_t size_; // size_ is updated in BlockBase.cpp in updateBlockStatistics methods
    size_t countOfNotNullValues_;
    size_t compressedSize_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;
    std::unique_ptr<int8_t[]> bitMask_;
    bool isCompressed_;
    bool isNullable_;
    bool wasRegistered_;
    bool isNullMaskRegistered_;
    bool saveNecessary_;

public:
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.BloclBase"/> class filled with data.
    /// </summary>
    /// <param name="data">Data which will fill up the block.</param>
    /// <param name="column">Column that will hold this new block.</param>
    BlockBase(const std::vector<T>& data, ColumnBase<T>& column, bool isCompressed = false, bool isNullable = false)
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
            int32_t bitMaskCapacity = ((capacity_ + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            bitMask_ = std::unique_ptr<int8_t[]>(new int8_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity);
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }
        std::copy(data.begin(), data.end(), data_.get());
        setBlockStatistics(data.size(), 0);
    }

    BlockBase(std::unique_ptr<T[]>&& data, int32_t dataSize, ColumnBase<T>& column, bool isCompressed = false, bool isNullable = false)
    : column_(column), size_(0), countOfNotNullValues_(0), isCompressed_(isCompressed),
      isNullable_(isNullable), bitMask_(nullptr), wasRegistered_(false),
      isNullMaskRegistered_(false), saveNecessary_(true)
    {
        capacity_ = (isCompressed) ? dataSize : column.GetBlockSize();
        data_ = std::unique_ptr<T[]>(new T[capacity_]);

        if (column_.GetBlockSize() < dataSize)
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }
        GPUMemory::hostPin(data_.get(), capacity_);
        wasRegistered_ = true;
        if (isNullable_)
        {
            int32_t bitMaskCapacity = ((capacity_ + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            bitMask_ = std::unique_ptr<int8_t[]>(new int8_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity);
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }
        data_ = std::move(data);

        setBlockStatistics(dataSize, 0);
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
            int32_t bitMaskCapacity = ((capacity_ + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            bitMask_ = std::unique_ptr<int8_t[]>(new int8_t[bitMaskCapacity]);
            std::memset(bitMask_.get(), 0, bitMaskCapacity);
            GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
            isNullMaskRegistered_ = true;
        }
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

    bool GetIsFullOfNullValue()
    {
        return isFullOfNullValue_;
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

    int8_t* GetNullBitmask()
    {
        return bitMask_.get();
    }

    size_t GetSize() const
    {
        return size_;
    }

    size_t GetNullBitmaskSize() const
    {
        return (size_ + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
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
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }
        std::copy(data.begin(), data.end(), data_.get() + size_);
        setBlockStatistics(data.size(), size_);
        saveNecessary_ = true;
    }

    void SetNullBitmask(const std::vector<int8_t>& nullMask)
    {
        if (isNullable_)
        {
            std::copy(nullMask.begin(), nullMask.end(), bitMask_.get());
            setBlockStatistics(0, 0);
        }
    }

    void SetNullBitmask(std::unique_ptr<int8_t[]>&& nullMask)
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
                int32_t bitMaskCapacity = ((capacity_ + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::hostPin(bitMask_.get(), bitMaskCapacity);
                isNullMaskRegistered_ = true;
            }
            setBlockStatistics(0, 0);
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
            data_.release();
            data_ = std::unique_ptr<T[]>(new T[capacity_]);

            GPUMemory::hostPin(data_.get(), capacity_);
            std::copy(dataCompressed.begin(), dataCompressed.end(), data_.get());

            isCompressed_ = true;
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
            data_.release();
            data_ = std::unique_ptr<T[]>(new T[capacity_]);

            GPUMemory::hostPin(data_.get(), capacity_);
            std::copy(dataDecompressed.begin(), dataDecompressed.end(), data_.get());

            isCompressed_ = false;
        }
    }

    /// <summary>
    /// Inserts data on proper position in block
    /// </summary>
    /// <param name="index">index in block where data will be inserted</param>
    /// <param name="data">value to insert<param>
    /// <param name="isNullValue">whether data is null value flag<param>
    void InsertDataOnSpecificPosition(int index, const T& data, bool isNullValue = false)
    {
        if (EmptyBlockSpace() == 0)
        {
            throw std::length_error("Attempted to insert data larger than remaining block size");
        }

        else if (index < size_)
        {
            std::move_backward(data_.get() + index, data_.get() + size_, data_.get() + size_ + 1);

            int bitMaskIdx = (index / (sizeof(char) * 8));
            int shiftIdx = (index % (sizeof(char) * 8));

            int last = isNullValue ? 1 : 0;

            if (isNullable_)
            {
                for (size_t i = shiftIdx; i < 8; i++)
                {
                    int tmp = (bitMask_[bitMaskIdx] >> i) & 1;

                    if (last != tmp)
                    {
                        if (last)
                        {
                            bitMask_[bitMaskIdx] |= (1 << i);
                        }
                        else
                        {
                            bitMask_[bitMaskIdx] &= ~(1 << i);
                        }

                        last = tmp;
                    }
                }

                bitMaskIdx++;
                int32_t bitMaskCapacity = ((capacity_ + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                for (size_t i = bitMaskIdx; i < bitMaskCapacity; i++)
                {
                    int tmp = bitMask_[i] >> 7;
                    bitMask_[i] <<= 1;
                    bitMask_[i] |= last;
                    last = tmp;
                }
            }
        }
        else if (isNullValue)
        {
            int bitMaskIdx = (index / (sizeof(char) * 8));
            int shiftIdx = (index % (sizeof(char) * 8));
            int last = isNullValue ? 1 : 0;
            bitMask_[bitMaskIdx] |= (last << shiftIdx);
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
