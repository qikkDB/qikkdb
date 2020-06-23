#include <numeric>
#include <algorithm>
#include <cmath>

#include "ColumnBase.h"
#include "PointFactory.h"
#include "Types/ComplexPolygon.pb.h"
#include "QueryEngine/NullConstants.cuh"

template <>
std::vector<int32_t> ColumnBase<int32_t>::NullArray(int length)
{
    return std::vector<int32_t>(length, GetNullConstant<int32_t>());
}

template <>
std::vector<float> ColumnBase<float>::NullArray(int length)
{
    return std::vector<float>(length, GetNullConstant<float>());
}

template <>
std::vector<int64_t> ColumnBase<int64_t>::NullArray(int length)
{
    return std::vector<int64_t>(length, GetNullConstant<int64_t>());
}

template <>
std::vector<double> ColumnBase<double>::NullArray(int length)
{
    return std::vector<double>(length, GetNullConstant<double>());
}

template <>
std::vector<int8_t> ColumnBase<int8_t>::NullArray(int length)
{
    return std::vector<int8_t>(length, GetNullConstant<int8_t>());
}

template <>
std::vector<std::string> ColumnBase<std::string>::NullArray(int length)
{
    return std::vector<std::string>(length, " ");
}

template <>
std::vector<ColmnarDB::Types::Point> ColumnBase<ColmnarDB::Types::Point>::NullArray(int length)
{
    return std::vector<ColmnarDB::Types::Point>(length, ColmnarDB::Types::Point());
}

template <>
std::vector<ColmnarDB::Types::ComplexPolygon> ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(int length)
{
    return std::vector<ColmnarDB::Types::ComplexPolygon>(length, ComplexPolygonFactory::FromWkt(
                                                                     "POLYGON((0 0, 0 0))"));
}

template <>
void ColumnBase<int32_t>::setColumnStatistics()
{
    std::vector<int32_t> mins;
    std::vector<int32_t> maxs;
    std::vector<int32_t> sums;

    std::vector<int64_t> numOfDataInBlocks;

    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
        sums.push_back(block->GetSum());
        numOfDataInBlocks.push_back(block->GetSize());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
    sum_ = std::accumulate(sums.begin(), sums.end(), 0);
    avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float)0.0);

    if (!initAvgIsSet_) // TODO spravit toto tak, aby sa nastavil initAvg_ az ked sa priemer vyrata z aspon X riadkov
    {
        initAvgIsSet_ = true;
        initAvg_ = avg_;
    }
}

template <>
void ColumnBase<int64_t>::setColumnStatistics()
{
    std::vector<int64_t> mins;
    std::vector<int64_t> maxs;
    std::vector<int64_t> sums;

    std::vector<int64_t> numOfDataInBlocks;

    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
        sums.push_back(block->GetSum());
        numOfDataInBlocks.push_back(block->GetSize());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
    sum_ = std::accumulate(sums.begin(), sums.end(), 0);
    avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float)0.0);
}

template <>
void ColumnBase<float>::setColumnStatistics()
{
    std::vector<float> mins;
    std::vector<float> maxs;
    std::vector<float> sums;

    std::vector<int64_t> numOfDataInBlocks;

    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
        sums.push_back(block->GetSum());
        numOfDataInBlocks.push_back(block->GetSize());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
    sum_ = std::accumulate(sums.begin(), sums.end(), (float)0.0);
    avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float)0.0);
}

template <>
void ColumnBase<double>::setColumnStatistics()
{
    std::vector<double> mins;
    std::vector<double> maxs;
    std::vector<double> sums;

    std::vector<int64_t> numOfDataInBlocks;

    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
        sums.push_back(block->GetSum());
        numOfDataInBlocks.push_back(block->GetSize());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
    sum_ = std::accumulate(sums.begin(), sums.end(), (double)0.0);
    avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float)0.0);
}

template <>
void ColumnBase<ColmnarDB::Types::Point>::setColumnStatistics()
{
    min_ = PointFactory::FromWkt("POINT(0 0)");
    max_ = PointFactory::FromWkt("POINT(0 0)");
    avg_ = (float)0.0;
    sum_ = PointFactory::FromWkt("POINT(0 0)");
}

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::setColumnStatistics()
{
    min_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0, 0 0))");
    max_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0, 0 0))");
    avg_ = (float)0.0;
    sum_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0, 0 0))");
}

template <>
void ColumnBase<std::string>::setColumnStatistics()
{
    avg_ = (float)0.0;
    sum_ = "";

    std::vector<std::string> mins;
    std::vector<std::string> maxs;


    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
}

template <>
void ColumnBase<int8_t>::setColumnStatistics()
{
    std::vector<int8_t> mins;
    std::vector<int8_t> maxs;
    std::vector<int8_t> sums;

    std::vector<int64_t> numOfDataInBlocks;

    for (auto& block : this->GetBlocksList())
    {
        mins.push_back(block->GetMin());
        maxs.push_back(block->GetMax());
        sums.push_back(block->GetSum());
        numOfDataInBlocks.push_back(block->GetSize());
    }

    min_ = *std::min_element(mins.begin(), mins.end());
    max_ = *std::max_element(maxs.begin(), maxs.end());
    sum_ = std::accumulate(sums.begin(), sums.end(), 0);
    avg_ = sum_ / std::accumulate(numOfDataInBlocks.begin(), numOfDataInBlocks.end(), (float)0.0);
}

template <>
void ColumnBase<std::string>::CopyDataToColumn(IColumn* destinationColumn)
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
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<int32_t> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    int32_t data;
                    try
                    {
                        data = std::stol(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = GetNullConstant<int32_t>();

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

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
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<int64_t> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    int64_t data;
                    try
                    {
                        data = std::stoll(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = GetNullConstant<int64_t>();

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

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
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<double> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    double data;
                    try
                    {
                        data = std::stod(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = GetNullConstant<double>();

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

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
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<float> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    float data;
                    try
                    {
                        data = std::stof(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = GetNullConstant<float>();

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

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
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<int8_t> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    int8_t data;

                    std::string upperStr = dataToCopy[j];
                    std::transform(dataToCopy[j].begin(), dataToCopy[j].end(), upperStr.begin(), ::toupper);

                    const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                    if (bit == static_cast<nullmask_t>(1U))
                    {
                        data = GetNullConstant<int8_t>();
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    else
                    {
                        if (upperStr == "TRUE")
                        {
                            data = 1;
                            NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                        }
                        else if (upperStr == "FALSE")
                        {
                            data = 0;
                            NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                        }
                        else
                        {
                            try
                            {
                                data = static_cast<int8_t>(std::stol(dataToCopy[j]));
                                data = data == 0 ? 0 : 1;
                                NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                            }
                            catch (std::invalid_argument)
                            {
                                data = GetNullConstant<int8_t>();
                                NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                            }
                        }
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));
                block.second.erase(block.second.begin());
            }
        }
    }
    break;

    case COLUMN_POINT:
    {
        auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(destinationColumn);

        for (auto& block : blocks_)
        {
            int32_t blockCountOnIdx = block.second.size();
            for (int32_t i = 0; i < blockCountOnIdx; i++)
            {
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<ColmnarDB::Types::Point> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    ColmnarDB::Types::Point data;
                    try
                    {
                        data = PointFactory::FromWkt(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = ColumnBase<ColmnarDB::Types::Point>::NullArray(1)[0];

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

                block.second.erase(block.second.begin());
            }
        }
    }
    break;

    case COLUMN_POLYGON:
    {
        auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(destinationColumn);

        for (auto& block : blocks_)
        {
            int32_t blockCountOnIdx = block.second.size();
            for (int32_t i = 0; i < blockCountOnIdx; i++)
            {
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<ColmnarDB::Types::ComplexPolygon> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    ColmnarDB::Types::ComplexPolygon data;
                    try
                    {
                        data = ComplexPolygonFactory::FromWkt(dataToCopy[j]);

                        const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                        NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                    }
                    catch (std::invalid_argument)
                    {
                        data = ColumnBase<ColmnarDB::Types::ComplexPolygon>::NullArray(1)[0];

                        NullValues::SetBitInBitMask(newNullMask.get(), j, static_cast<nullmask_t>(1U));
                    }
                    castedDataToCopy.push_back(data);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

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

template <>
void ColumnBase<ColmnarDB::Types::Point>::CopyDataToColumn(IColumn* destinationColumn)
{
    auto toType = destinationColumn->GetColumnType();

    if (toType == COLUMN_STRING)
    {
        auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(destinationColumn);

        for (auto& block : blocks_)
        {
            int32_t blockCountOnIdx = block.second.size();
            for (int32_t i = 0; i < blockCountOnIdx; i++)
            {
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<std::string> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    std::string data = PointFactory::WktFromPoint(dataToCopy[j]);
                    castedDataToCopy.push_back(data);

                    const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                    NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

                block.second.erase(block.second.begin());
            }
        }
    }

    else
	{
        throw std::runtime_error("Attempt to execute unsupported column type conversion.");
    }
}

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::CopyDataToColumn(IColumn* destinationColumn)
{
    auto toType = destinationColumn->GetColumnType();

    if (toType == COLUMN_STRING)
    {
        auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(destinationColumn);

        for (auto& block : blocks_)
        {
            int32_t blockCountOnIdx = block.second.size();
            for (int32_t i = 0; i < blockCountOnIdx; i++)
            {
                nullmask_t* mask = block.second.front()->GetNullBitmask();

                auto dataToCopy = block.second.front()->GetData();
                auto blockSize = block.second.front()->GetSize();
                std::vector<std::string> castedDataToCopy;
                std::unique_ptr<nullmask_t[]> newNullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[NullValues::GetNullBitMaskSize(blockSize)]);
                std::memset(newNullMask.get(), 0, NullValues::GetNullBitMaskSize(blockSize) * sizeof(nullmask_t));

                for (int32_t j = 0; j < blockSize; j++)
                {
                    std::string data = ComplexPolygonFactory::WktFromPolygon(dataToCopy[j]);
                    castedDataToCopy.push_back(data);

                    const nullmask_t bit = NullValues::GetConcreteBitFromBitmask(mask, j);
                    NullValues::SetBitInBitMask(newNullMask.get(), j, bit);
                }

                auto& newBlock = castedColumn->AddBlock(castedDataToCopy, block.first);
                newBlock.SetNullBitmask(std::move(newNullMask));

                block.second.erase(block.second.begin());
            }
        }
    }

    else
    {
        throw std::runtime_error("Attempt to execute unsupported column type conversion.");
    }
}

template <>
void ColumnBase<std::string>::SetIsUnique(bool isUnique)
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

        std::string duplicateData;
        bool duplicateFound = false;
        for (auto const& mapBlock : blocks_)
        {
            for (int32_t i = 0; i < mapBlock.second.size() && !duplicateFound; i++)
            {
                auto data = mapBlock.second[i]->GetData();

                for (int32_t j = 0; j < mapBlock.second[i]->GetSize() && !duplicateFound; j++)
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
            uniqueHashmap_.clear();
            throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                                             "Could not add UNIQUE constraint on column: " + name_ +
                                                 ", column contains duplicate value: " + duplicateData);
        }
    }
    else
    {
        isUnique_ = false;
        BOOST_LOG_TRIVIAL(debug) << "Flag isUnique_ was set to FALSE for column named: " << name_ << ".";
    }
}

template <>
void ColumnBase<ColmnarDB::Types::Point>::SetIsUnique(bool isUnique)
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

        ColmnarDB::Types::Point duplicateData;
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
            uniqueHashmap_.clear();
            throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                                             "Could not add UNIQUE constraint on column: " + name_ +
                                                 ", column contains duplicate value: " +
                                                 PointFactory::WktFromPoint(duplicateData));
        }
    }
    else
    {
        isUnique_ = false;
        BOOST_LOG_TRIVIAL(debug) << "Flag isUnique_ was set to FALSE for column named: " << name_ << ".";
    }
}

template <>
void ColumnBase<ColmnarDB::Types::ComplexPolygon>::SetIsUnique(bool isUnique)
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

        ColmnarDB::Types::ComplexPolygon duplicateData;
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
            uniqueHashmap_.clear();
            throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                                             "Could not add UNIQUE constraint on column: " + name_ +
                                                 ", column contains duplicate value: " +
                                                 ComplexPolygonFactory::WktFromPolygon(duplicateData));
        }
    }
    else
    {
        isUnique_ = false;
        BOOST_LOG_TRIVIAL(debug) << "Flag isUnique_ was set to FALSE for column named: " << name_ << ".";
    }
}