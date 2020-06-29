#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"

#include <numeric>
#include <algorithm>
#include <cmath>

/// <summary>
/// Update block statistics.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<int32_t>::updateBlockStatistics(const int32_t& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = data;
        sum_ = data;
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
        sum_ += data;
        avg_ = sum_ / static_cast<float>(countOfNotNullValues_);
    }
}

/// <summary>
/// Update block statistics.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<int64_t>::updateBlockStatistics(const int64_t& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = data;
        sum_ = data;
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
        sum_ += data;
        avg_ = sum_ / static_cast<float>(countOfNotNullValues_);
    }
}

/// <summary>
/// Update block statistics.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<float>::updateBlockStatistics(const float& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = data;
        sum_ = data;
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
        sum_ += data;
        avg_ = sum_ / static_cast<float>(countOfNotNullValues_);
    }
}

/// <summary>
/// Update block statistics.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<double>::updateBlockStatistics(const double& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = data;
        sum_ = data;
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
        sum_ += data;
        avg_ = sum_ / static_cast<float>(countOfNotNullValues_);
    }
}

/// <summary>
/// Update block statistics. For point are min, max and sum set to default point value and avg is 0.0.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<QikkDB::Types::Point>::updateBlockStatistics(const QikkDB::Types::Point& data, bool isNullValue)
{
    if (size_ == 0)
    {
        min_ = PointFactory::FromWkt("POINT(0 0)");
        max_ = PointFactory::FromWkt("POINT(0 0)");
        avg_ = (float)0.0;
        sum_ = PointFactory::FromWkt("POINT(0 0)");
    }

    size_++;
}

/// <summary>
/// Update block statistics. For polygon are min, max and sum set to default polygon value and avg is 0.0.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<QikkDB::Types::ComplexPolygon>::updateBlockStatistics(const QikkDB::Types::ComplexPolygon& data,
                                                                        bool isNullValue)
{
    if (size_ == 0)
    {
        min_ = ComplexPolygonFactory::FromWkt(ColumnBase<QikkDB::Types::ComplexPolygon>::POLYGON_DEFAULT_VALUE);
        max_ = ComplexPolygonFactory::FromWkt(ColumnBase<QikkDB::Types::ComplexPolygon>::POLYGON_DEFAULT_VALUE);
        avg_ = (float)0.0;
        sum_ = ComplexPolygonFactory::FromWkt(ColumnBase<QikkDB::Types::ComplexPolygon>::POLYGON_DEFAULT_VALUE);
    }

    size_++;
}

/// <summary>
/// Update block statistics. For string is sum "" and avg 0.0.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<std::string>::updateBlockStatistics(const std::string& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = (float)0.0;
        sum_ = "";
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
    }
}

/// <summary>
/// Update block statistics.
/// </summary>
/// <param name="data">Data that are inserted into block.</param>
/// <param name="isNullValue">Flag indicating whether the inserted data is null value.</param>
template <>
void BlockBase<int8_t>::updateBlockStatistics(const int8_t& data, bool isNullValue)
{
    if (isCompressed_)
    {
        return;
    }

    if (isNullValue)
    {
        size_++;
        return;
    }

    if (size_ == 0)
    {
        size_++;
        countOfNotNullValues_++;

        min_ = data;
        max_ = data;
        avg_ = data;
        sum_ = data;
    }
    else
    {
        size_++;
        countOfNotNullValues_++;

        min_ = (data < min_) ? data : min_;
        max_ = (data > max_) ? data : max_;
        sum_ += data;
        avg_ = sum_ / static_cast<float>(countOfNotNullValues_);
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<int32_t>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    if (isCompressed_)
    {
        return;
    }

    if (!isNullable_)
    {
        for (int64_t i = 0; i < insertedDataSize; i++)
        {
            updateBlockStatistics(data_[i + oldDataSize], false);
        }
    }
    else
    {
        size_ += insertedDataSize;

        min_ = std::numeric_limits<int32_t>::max();
        max_ = std::numeric_limits<int32_t>::lowest();
        avg_ = 0;
        sum_ = 0;
        size_t count = 0;
        countOfNotNullValues_ = 0;
        for (int64_t i = 0; i < size_; i++)
        {
            nullmask_t isNull = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), i);
            if (!isNull)
            {
                countOfNotNullValues_++;

                min_ = std::min(min_, data_[i]);
                max_ = std::max(max_, data_[i]);
                sum_ += data_[i];
                avg_ += data_[i];
                count++;
            }
        }
        avg_ /= count;
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<int64_t>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    if (isCompressed_)
    {
        return;
    }

    if (!isNullable_)
    {

        for (int64_t i = 0; i < insertedDataSize; i++)
        {
            updateBlockStatistics(data_[i + oldDataSize], false);
        }
    }
    else
    {
        size_ += insertedDataSize;

        min_ = std::numeric_limits<int64_t>::max();
        max_ = std::numeric_limits<int64_t>::lowest();
        avg_ = 0;
        sum_ = 0;
        size_t count = 0;
        countOfNotNullValues_ = 0;
        for (int64_t i = 0; i < size_; i++)
        {
            nullmask_t isNull = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), i);
            if (!isNull)
            {
                countOfNotNullValues_++;

                min_ = std::min(min_, data_[i]);
                max_ = std::max(max_, data_[i]);
                sum_ += data_[i];
                avg_ += data_[i];
                count++;
            }
        }
        avg_ /= count;
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<float>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    if (isCompressed_)
    {
        return;
    }

    if (!isNullable_)
    {
        for (int64_t i = 0; i < insertedDataSize; i++)
        {
            updateBlockStatistics(data_[i + oldDataSize], false);
        }
    }
    else
    {
        size_ += insertedDataSize;

        min_ = std::numeric_limits<float>::max();
        max_ = std::numeric_limits<float>::lowest();
        avg_ = 0;
        sum_ = 0;
        size_t count = 0;
        countOfNotNullValues_ = 0;
        for (int64_t i = 0; i < size_; i++)
        {
            nullmask_t isNull = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), i);
            if (!isNull)
            {
                countOfNotNullValues_++;

                min_ = std::min(min_, data_[i]);
                max_ = std::max(max_, data_[i]);
                sum_ += data_[i];
                avg_ += data_[i];
                count++;
            }
        }
        avg_ /= count;
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<double>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    if (isCompressed_)
    {
        return;
    }
    if (!isNullable_)
    {
        for (int64_t i = 0; i < insertedDataSize; i++)
        {
            updateBlockStatistics(data_[i + oldDataSize], false);
        }
    }
    else
    {
        size_ += insertedDataSize;

        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
        avg_ = 0;
        sum_ = 0;
        size_t count = 0;
        countOfNotNullValues_ = 0;
        for (int64_t i = 0; i < size_; i++)
        {
            nullmask_t isNull = NullValues::GetConcreteBitFromBitmask(bitMask_.get(), i);
            if (!isNull)
            {
                countOfNotNullValues_++;

                min_ = std::min(min_, data_[i]);
                max_ = std::max(max_, data_[i]);
                sum_ += data_[i];
                avg_ += data_[i];
                count++;
            }
        }
        avg_ /= count;
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<QikkDB::Types::Point>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    for (int64_t i = 0; i < insertedDataSize; i++)
    {
        updateBlockStatistics(data_[i + oldDataSize], false);
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<QikkDB::Types::ComplexPolygon>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    for (int64_t i = 0; i < insertedDataSize; i++)
    {
        updateBlockStatistics(data_[i + oldDataSize], false);
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<std::string>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    for (int64_t i = 0; i < insertedDataSize; i++)
    {
        updateBlockStatistics(data_[i + oldDataSize], false);
    }
}

/// <summary>
/// Set Block Statistics
/// </summary>
/// <param name="insertedDataSize">represents size of data which are inserted</param>
/// <param name="oldDataSize">represents size of data which are already inserted in block</param>
template <>
void BlockBase<int8_t>::setBlockStatistics(int32_t insertedDataSize, int32_t oldDataSize)
{
    if (isCompressed_)
    {
        return;
    }

    for (int64_t i = 0; i < insertedDataSize; i++)
    {
        updateBlockStatistics(data_[i + oldDataSize], false);
    }
}