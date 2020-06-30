#include <boost/log/trivial.hpp>
#include <cstdint>

#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include "QueryEngine/NullConstants.cuh"
#include "ConstraintViolationError.h"


#ifndef __CUDACC__
/// <summary>
/// Insert row of data to database on specific position
/// </summary>
/// <param name="data">column name with inserting data</param>
/// <param name="indexBlock">index of block where data will be inserted</param>
/// <param name="indexInBlock">index in block where data will be inserted</param>
/// <param name="iterator">index of row of data</param>
/// <param name="nullMask">column name with bitmask</param>
void Table::InsertValuesOnSpecificPosition(const std::unordered_map<std::string, std::any>& data,
                                           const int32_t indexBlock,
                                           const int32_t indexInBlock,
                                           const int32_t iterator,
                                           const std::unordered_map<std::string, std::vector<nullmask_t>>& nullMasks)
{
    for (const auto& column : columns_)
    {
        const std::string columnName = column.first;
        auto currentColumn = (columns_.find(columnName)->second.get());
        if (data.find(columnName) != data.end())
        {
            nullmask_t isNullValue = static_cast<nullmask_t>(0U);
            if (nullMasks.find(columnName) != nullMasks.end())
            {
                isNullValue = NullValues::GetConcreteBitFromBitmask(nullMasks.at(columnName).data(), iterator);
            }

            const auto& wrappedData = data.at(columnName);

            if (wrappedData.type() == typeid(std::vector<int32_t>))
            {
                std::vector<int32_t> dataInt = std::any_cast<std::vector<int32_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataInt[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataInt[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<int64_t>))
            {
                std::vector<int64_t> dataLong = std::any_cast<std::vector<int64_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataLong[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataLong[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<double>))
            {
                std::vector<double> dataDouble = std::any_cast<std::vector<double>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataDouble[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataDouble[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<float>))
            {
                std::vector<float> dataFloat = std::any_cast<std::vector<float>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataFloat[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataFloat[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::Point>))
            {
                std::vector<QikkDB::Types::Point> dataPoint =
                    std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataPoint[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataPoint[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::ComplexPolygon>))
            {
                std::vector<QikkDB::Types::ComplexPolygon> dataPolygon =
                    std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataPolygon[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataPolygon[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<std::string>))
            {
                std::vector<std::string> dataString = std::any_cast<std::vector<std::string>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataString[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataString[iterator]);
                }
            }
            else if (wrappedData.type() == typeid(std::vector<int8_t>))
            {
                std::vector<int8_t> dataBool = std::any_cast<std::vector<int8_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int8_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock,
                                                           dataBool[iterator], -1, isNullValue);
                if (castedColumn->GetIsUnique())
                {
                    castedColumn->InsertIntoHashmap(dataBool[iterator]);
                }
            }
        }
    }
}

/// <summary>
/// Gets count ofrows of inserted  data
/// </summary>
/// <param name="data">unordered map of columnName and data that should be inserted in this
/// column</param> <returns>Count of rows of inserted data</returns>
int32_t Table::GetDataSizeOfInsertedColumns(const std::unordered_map<std::string, std::any>& data)
{
    int size = 0;
    const auto& dataOfFirstColumn = data.begin()->second;

    if (dataOfFirstColumn.type() == typeid(std::vector<int32_t>))
    {
        std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

    else if (dataOfFirstColumn.type() == typeid(std::vector<int64_t>))
    {
        std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

    else if (dataOfFirstColumn.type() == typeid(std::vector<double>))
    {
        std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

    else if (dataOfFirstColumn.type() == typeid(std::vector<float>))
    {
        std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

    else if (dataOfFirstColumn.type() == typeid(std::vector<std::string>))
    {
        std::vector<std::string> dataIndexedColumn = std::any_cast<std::vector<std::string>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }
    else if (dataOfFirstColumn.type() == typeid(std::vector<QikkDB::Types::Point>))
    {
        std::vector<QikkDB::Types::Point> dataIndexedColumn =
            std::any_cast<std::vector<QikkDB::Types::Point>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }
    else if (dataOfFirstColumn.type() == typeid(std::vector<QikkDB::Types::ComplexPolygon>))
    {
        std::vector<QikkDB::Types::ComplexPolygon> dataIndexedColumn =
            std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }
    else if (dataOfFirstColumn.type() == typeid(std::vector<int8_t>))
    {
        std::vector<int8_t> dataIndexedColumn = std::any_cast<std::vector<int8_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

    return size;
}

/// <summary>
/// Checks UNIQUE constraints in table and data which should be inserted if fit this requirements
/// </summary>
/// <param name="data">unordered map of columnName and data that should be inserted in this
/// <param name="mullMasks">unordered map of columnName and vector which represents null values
/// <param name="dataSize">num of data in one column
void Table::CheckUniqueConstraintInData(const std::unordered_map<std::string, std::any>& data, int32_t dataSize)
{
    bool duplicateFound = false;
    std::string nameOfUniqueColumn;

    for (const auto& column : columns_)
    {
        if (data.find(column.first) != data.end())
        {
            if (column.second.get()->GetIsUnique() && !duplicateFound)
            {
                DataType columnType = column.second.get()->GetColumnType();

                switch (columnType)
                {

                case COLUMN_INT:
                {
                    int32_t duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(column.second.get());
                    std::unordered_set<int32_t> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<int32_t> dataColumn =
                        std::any_cast<std::vector<int32_t>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }

                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn +
                                ", column contains duplicate value: " + std::to_string(duplicateData));
                    }
                }
                break;

                case COLUMN_LONG:
                {
                    int64_t duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(column.second.get());
                    std::unordered_set<int64_t> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<int64_t> dataColumn =
                        std::any_cast<std::vector<int64_t>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }

                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn +
                                ", column contains duplicate value: " + std::to_string(duplicateData));
                    }
                }
                break;

                case COLUMN_DOUBLE:
                {
                    double duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<double>*>(column.second.get());
                    std::unordered_set<double> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<double> dataColumn =
                        std::any_cast<std::vector<double>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn +
                                ", column contains duplicate value: " + std::to_string(duplicateData));
                    }
                }
                break;

                case COLUMN_FLOAT:
                {
                    float duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<float>*>(column.second.get());
                    std::unordered_set<float> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<float> dataColumn = std::any_cast<std::vector<float>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn +
                                ", column contains duplicate value: " + std::to_string(duplicateData));
                    }
                }
                break;

                case COLUMN_POINT:
                {
                    QikkDB::Types::Point duplicateData;
                    auto castedColumn =
                        dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(column.second.get());
                    std::unordered_set<QikkDB::Types::Point> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<QikkDB::Types::Point> dataColumn =
                        std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn + ", column contains duplicate value: " +
                                PointFactory::WktFromPoint(duplicateData));
                    }
                }
                break;

                case COLUMN_POLYGON:
                {
                    QikkDB::Types::ComplexPolygon duplicateData;
                    auto castedColumn =
                        dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(column.second.get());
                    std::unordered_set<QikkDB::Types::ComplexPolygon> temp_hashmap =
                        castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<QikkDB::Types::ComplexPolygon> dataColumn =
                        std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn + ", column contains duplicate value: " +
                                ComplexPolygonFactory::WktFromPolygon(duplicateData));
                    }
                }
                break;

                case COLUMN_STRING:
                {
                    std::string duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(column.second.get());
                    std::unordered_set<std::string> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<std::string> dataColumn =
                        std::any_cast<std::vector<std::string>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn + ", column contains duplicate value: " + duplicateData);
                    }
                }
                break;

                case COLUMN_INT8_T:
                {
                    int8_t duplicateData;
                    auto castedColumn = dynamic_cast<ColumnBase<int8_t>*>(column.second.get());
                    std::unordered_set<int8_t> temp_hashmap = castedColumn->GetHashmapCopy();

                    const auto& wrappedCurrentColumnData = data.at(column.first);
                    std::vector<int8_t> dataColumn =
                        std::any_cast<std::vector<int8_t>>(wrappedCurrentColumnData);

                    for (int i = 0; i < dataSize && !duplicateFound; i++)
                    {
                        if (castedColumn->IsDuplicate(temp_hashmap, dataColumn[i]))
                        {
                            duplicateFound = true;
                            nameOfUniqueColumn = column.first;
                            duplicateData = dataColumn[i];
                        }
                        else
                        {
                            temp_hashmap.insert(dataColumn[i]);
                        }
                    }
                    if (duplicateFound)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
                            "Could not execute statement due to a UNIQUE "
                            "constraint failure on column: " +
                                nameOfUniqueColumn +
                                ", column contains duplicate value: " + std::to_string(duplicateData));
                    }
                }
                break;
                default:
                    throw std::runtime_error("Unsupported column type.");
                    break;
                }
            }
        }
        else
        {
            // Case when column is not in inserted data so automaticaly there should be NULL inserted which is not suitable when column is Unique
            if (column.second.get()->GetIsUnique())
            {
                throw constraint_violation_error(ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
                                                 "Could not execute statement due to a UNIQUE "
                                                 "constraint failure on column: " +
                                                     column.first + "trying to insert NULL value");
            }
        }
    }
}

void Table::CheckNullableConstraintInData(const std::unordered_map<std::string, std::vector<nullmask_t>>& nullMasks,
                                          int32_t dataSize)
{
    bool nullValueFound = false;
    std::string nameOfColumn;

    for (const auto& column : columns_)
    {
        if (nullMasks.find(column.first) != nullMasks.end())
        {
            if (!column.second.get()->GetIsNullable() && !nullValueFound)
            {

                for (int i = 0; i < dataSize && !nullValueFound; i++)
                {
                    nullmask_t isNullValue = false;
                    if (nullMasks.find(column.first) != nullMasks.end())
                    {
                        isNullValue =
                            NullValues::GetConcreteBitFromBitmask(nullMasks.at(column.first).data(), i);
                    }

                    if (isNullValue)
                    {
                        throw constraint_violation_error(
                            ConstraintViolationErrorType::UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
                            "Could not execute statement due to a NOT NULL "
                            "constraint failure on column: " +
                                column.first + " trying to insert NULL value");
                    }
                }
            }
        }
    }
}

bool Table::GetHasUniqueConstraints()
{
    for (const auto& column : columns_)
    {
        if (column.second.get()->GetIsUnique())
        {
            return true;
        }
    }
    return false;
}

bool Table::GetHasNotNullConstraints()
{
    for (const auto& column : columns_)
    {
        if (!column.second.get()->GetIsNullable())
        {
            return true;
        }
    }
    return false;
}

/// <summary>
/// Gets count of rows that are already inserted in database - information is getting as count of data in first sorting column
/// </summary>
/// <returns>count of rows of data in database</returns>
int32_t Table::GetDataRangeInSortingColumn()
{
    int size = 0;

    auto firstSortingColumn = (columns_.find(sortingColumns_[0])->second.get());
    auto columnType = firstSortingColumn->GetColumnType();

    if (columnType == COLUMN_INT)
    {
        auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();
        int blockCount = castedColumn->GetBlockCount();

        for (int i = 0; i < blockCount; i++)
        {
            size += blocks[i]->GetSize();
        }
    }

    else if (columnType == COLUMN_LONG)
    {
        auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();
        int blockCount = castedColumn->GetBlockCount();

        for (int i = 0; i < blockCount; i++)
        {
            size += blocks[i]->GetSize();
        }
    }

    else if (columnType == COLUMN_DOUBLE)
    {
        auto castedColumn = dynamic_cast<ColumnBase<double>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();
        int blockCount = castedColumn->GetBlockCount();

        for (int i = 0; i < blockCount; i++)
        {
            size += blocks[i]->GetSize();
        }
    }

    else if (columnType == COLUMN_FLOAT)
    {
        auto castedColumn = dynamic_cast<ColumnBase<float>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();
        int blockCount = castedColumn->GetBlockCount();

        for (int i = 0; i < blockCount; i++)
        {
            size += blocks[i]->GetSize();
        }
    }

    else if (columnType == COLUMN_STRING)
    {
        auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();
        int blockCount = castedColumn->GetBlockCount();

        for (int i = 0; i < blockCount; i++)
        {
            size += blocks[i]->GetSize();
        }
    }
    return size;
}

/// <summary>
/// Gets one row and bitmask of this row from specific position from inserted data
/// </summary>
/// <param name="data">name of column with inserted data</param>
/// <param name="iterator">position of row to get</param>
/// <param name="nullMask">name of column with bitmasks of these data</param>
/// <returns>tuple of row and bitmask of this row from specific position</returns>
std::tuple<std::vector<std::any>, std::vector<uint8_t>>
Table::GetRowAndBitmaskOfInsertedData(const std::unordered_map<std::string, std::any>& data,
                                      const int32_t iterator,
                                      const std::unordered_map<std::string, std::vector<nullmask_t>>& nullMasks)
{
    std::vector<std::any> resultRow;
    std::vector<uint8_t> maskOfRow;

    for (auto column : sortingColumns_)
    {
        int8_t isNullValue = 0;
        if (nullMasks.find(column) != nullMasks.end())
        {
            isNullValue = static_cast<int8_t>(
                NullValues::GetConcreteBitFromBitmask(nullMasks.at(column).data(), iterator));
        }

        maskOfRow.push_back(isNullValue);

        if (data.find(column) != data.end())
        {
            const auto& wrappedCurrentSortingColumnData = data.at(column);

            if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int32_t>))
            {
                std::vector<int32_t> dataIndexedColumn =
                    std::any_cast<std::vector<int32_t>>(wrappedCurrentSortingColumnData);
                resultRow.push_back(dataIndexedColumn[iterator]);
            }

            else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int64_t>))
            {
                std::vector<int64_t> dataIndexedColumn =
                    std::any_cast<std::vector<int64_t>>(wrappedCurrentSortingColumnData);
                resultRow.push_back(dataIndexedColumn[iterator]);
            }

            else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<float>))
            {
                std::vector<float> dataIndexedColumn =
                    std::any_cast<std::vector<float>>(wrappedCurrentSortingColumnData);
                resultRow.push_back(dataIndexedColumn[iterator]);
            }

            else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<double>))
            {
                std::vector<double> dataIndexedColumn =
                    std::any_cast<std::vector<double>>(wrappedCurrentSortingColumnData);
                resultRow.push_back(dataIndexedColumn[iterator]);
            }

            else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<std::string>))
            {
                std::vector<std::string> dataIndexedColumn =
                    std::any_cast<std::vector<std::string>>(wrappedCurrentSortingColumnData);
                resultRow.push_back(dataIndexedColumn[iterator]);
            }
        }
        else
        {
            switch (columns_.at(column)->GetColumnType())
            {
            case COLUMN_INT:
                resultRow.push_back(GetNullConstant<int32_t>());
                break;

            case COLUMN_LONG:
                resultRow.push_back(GetNullConstant<int64_t>());
                break;

            case COLUMN_FLOAT:
                resultRow.push_back(GetNullConstant<float>());
                break;

            case COLUMN_DOUBLE:
                resultRow.push_back(GetNullConstant<double>());
                break;

            case COLUMN_INT8_T:
                resultRow.push_back(GetNullConstant<int8_t>());
                break;

            case COLUMN_STRING:
                resultRow.push_back(ColumnBase<std::string>::NullArray(1)[0]);
                break;

            case COLUMN_POINT:
                resultRow.push_back(ColumnBase<QikkDB::Types::Point>::NullArray(1)[0]);
                break;

            case COLUMN_POLYGON:
                resultRow.push_back(ColumnBase<QikkDB::Types::ComplexPolygon>::NullArray(1)[0]);
                break;
            default:
                throw std::runtime_error("Unsupported column type.");
                break;
            }
        }
    }
    return std::make_tuple(resultRow, maskOfRow);
}

/// <summary>
/// Gets block index and index in block from total index to database
/// </summary>
/// <param name="index">index to database</param>
/// <param name="positionToCompare">whether method is used to get indices to compare row from these
/// indices or we're looking for indices to insert row</param> <returns>block index, index in
/// block</returns>
std::tuple<int32_t, int32_t> Table::GetIndicesFromTotalIndex(const int32_t index, bool positionToCompare)
{
    int32_t blockIndex = 0;
    int32_t indexInBlock = index;

    auto firstSortingColumn = (columns_.find(sortingColumns_[0])->second.get());
    auto columnType = firstSortingColumn->GetColumnType();

    int32_t positionDiff = positionToCompare ? 1 : 0;

    if (columnType == COLUMN_INT)
    {
        auto castedColumn = dynamic_cast<const ColumnBase<int32_t>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();

        int i = 0;

        while (static_cast<int64_t>(indexInBlock - blocks[i]->GetSize() + positionDiff) > 0)
        {
            indexInBlock -= blocks[i]->GetSize();
            blockIndex++;

            i++;
        }
    }

    else if (columnType == COLUMN_LONG)
    {
        auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();

        int i = 0;

        while (static_cast<int64_t>(indexInBlock - blocks[i]->GetSize() + positionDiff) > 0)
        {
            indexInBlock -= blocks[i]->GetSize();
            blockIndex++;

            i++;
        }
    }

    else if (columnType == COLUMN_DOUBLE)
    {
        auto castedColumn = dynamic_cast<ColumnBase<double>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();

        int i = 0;

        while (static_cast<int64_t>(indexInBlock - blocks[i]->GetSize() + positionDiff) > 0)
        {
            indexInBlock -= blocks[i]->GetSize();
            blockIndex++;

            i++;
        }
    }

    else if (columnType == COLUMN_FLOAT)
    {
        auto castedColumn = dynamic_cast<ColumnBase<float>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();

        int i = 0;

        while (static_cast<int64_t>(indexInBlock - blocks[i]->GetSize() + positionDiff) > 0)
        {
            indexInBlock -= blocks[i]->GetSize();
            blockIndex++;

            i++;
        }
    }

    else if (columnType == COLUMN_STRING)
    {
        auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(firstSortingColumn);
        auto& blocks = castedColumn->GetBlocksList();

        int i = 0;

        while (static_cast<int64_t>(indexInBlock - blocks[i]->GetSize() + positionDiff) > 0)
        {
            indexInBlock -= blocks[i]->GetSize();
            blockIndex++;

            i++;
        }
    }
    return std::make_tuple(blockIndex, indexInBlock);
}

/// <summary>
/// Gets row and its bitmask from database from specific position
/// </summary>
/// <param name="index">index to database where row should be extracted from</param>
/// <returns>values of row and its bitmask</returns>
std::tuple<std::vector<std::any>, std::vector<int8_t>> Table::GetRowAndBitmaskOnIndex(const int32_t index)
{
    int32_t blockIndex;
    int32_t indexInBlock;

    std::tie(blockIndex, indexInBlock) = GetIndicesFromTotalIndex(index, true);
    std::vector<std::any> resultRow;
    std::vector<int8_t> maskOfRow;

    int8_t isNullValue = 0;

    for (auto sortingColumn : sortingColumns_)
    {
        auto currentSortingColumn = (columns_.find(sortingColumn)->second.get());
        auto columnType = currentSortingColumn->GetColumnType();

        if (columnType == COLUMN_INT)
        {
            auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);
            resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

            isNullValue = static_cast<int8_t>(NullValues::GetConcreteBitFromBitmask(
                castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask(), indexInBlock));
            maskOfRow.push_back(isNullValue);
        }

        else if (columnType == COLUMN_LONG)
        {
            auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentSortingColumn);
            resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

            isNullValue = static_cast<int8_t>(NullValues::GetConcreteBitFromBitmask(
                castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask(), indexInBlock));
            maskOfRow.push_back(isNullValue);
        }

        else if (columnType == COLUMN_FLOAT)
        {
            auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentSortingColumn);
            resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

            isNullValue = static_cast<int8_t>(NullValues::GetConcreteBitFromBitmask(
                castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask(), indexInBlock));
            maskOfRow.push_back(isNullValue);
        }

        else if (columnType == COLUMN_DOUBLE)
        {
            auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentSortingColumn);
            resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

            isNullValue = static_cast<int8_t>(NullValues::GetConcreteBitFromBitmask(
                castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask(), indexInBlock));
            maskOfRow.push_back(isNullValue);
        }

        else if (columnType == COLUMN_STRING)
        {
            auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(currentSortingColumn);
            resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

            isNullValue = static_cast<int8_t>(NullValues::GetConcreteBitFromBitmask(
                castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask(), indexInBlock));
            maskOfRow.push_back(isNullValue);
        }
    }
    return std::make_tuple(resultRow, maskOfRow);
}

/// <summary>
/// Compare two rows (one that should be inserted and one from database) of data according their values and bitmasks
/// </summary>
/// <param name="rowToInsert">values of inserted row of data</param>
/// <param name="maskOfInsertedRow">bitmask of inserted row</param>
/// <param name="index">index of row in database that should be compare with inserted row</param>
/// <returns>one of enum value - Greater, Lower, Equal - according to relationship of inserted row and row from database</returns>
Table::CompareResult
Table::CompareRows(std::vector<std::any> rowToInsert, std::vector<uint8_t> maskOfInsertRow, const int32_t index)
{
    std::vector<std::any> rowToCompare;
    std::vector<int8_t> maskOfCompareRow;

    std::tie(rowToCompare, maskOfCompareRow) = GetRowAndBitmaskOnIndex(index);

    for (int i = 0; i < sortingColumns_.size(); i++)
    {
        int8_t insertBit = maskOfInsertRow[i];
        int8_t compareBit = maskOfCompareRow[i];

        if (insertBit == 1 && compareBit == 0)
        {
            return CompareResult::Lower;
        }

        else if (insertBit == 0 && compareBit == 1)
        {
            return CompareResult::Greater;
        }

        else if (insertBit == 0 && compareBit == 0)
        {
            if ((columns_.find(sortingColumns_[i])->second.get())->GetColumnType() == COLUMN_INT)
            {
                if (std::any_cast<int32_t>(rowToInsert[i]) < std::any_cast<int32_t>(rowToCompare[i]))
                {
                    return CompareResult::Lower;
                }

                else if (std::any_cast<int32_t>(rowToInsert[i]) > std::any_cast<int32_t>(rowToCompare[i]))
                {
                    return CompareResult::Greater;
                }
            }

            else if ((columns_.find(sortingColumns_[i])->second.get())->GetColumnType() == COLUMN_LONG)
            {
                if (std::any_cast<int64_t>(rowToInsert[i]) < std::any_cast<int64_t>(rowToCompare[i]))
                {
                    return CompareResult::Lower;
                }

                else if (std::any_cast<int64_t>(rowToInsert[i]) > std::any_cast<int64_t>(rowToCompare[i]))
                {
                    return CompareResult::Greater;
                }
            }

            else if ((columns_.find(sortingColumns_[i])->second.get())->GetColumnType() == COLUMN_DOUBLE)
            {
                if (std::any_cast<double>(rowToInsert[i]) < std::any_cast<double>(rowToCompare[i]))
                {
                    return CompareResult::Lower;
                }

                else if (std::any_cast<double>(rowToInsert[i]) > std::any_cast<double>(rowToCompare[i]))
                {
                    return CompareResult::Greater;
                }
            }

            else if ((columns_.find(sortingColumns_[i])->second.get())->GetColumnType() == COLUMN_FLOAT)
            {
                if (std::any_cast<float>(rowToInsert[i]) < std::any_cast<float>(rowToCompare[i]))
                {
                    return CompareResult::Lower;
                }

                else if (std::any_cast<float>(rowToInsert[i]) > std::any_cast<float>(rowToCompare[i]))
                {
                    return CompareResult::Greater;
                }
            }

            else if ((columns_.find(sortingColumns_[i])->second.get())->GetColumnType() == COLUMN_STRING)
            {
                if (std::any_cast<std::string>(rowToInsert[i]) < std::any_cast<std::string>(rowToCompare[i]))
                {
                    return CompareResult::Lower;
                }

                else if (std::any_cast<std::string>(rowToInsert[i]) >
                         std::any_cast<std::string>(rowToCompare[i]))
                {
                    return CompareResult::Greater;
                }
            }
        }
    }
    return CompareResult::Equal;
}

/// <summary>
/// Gets indices of block and position in block where should be row of data inserted according to its values and bitmask
/// </summary>
/// <param name="rowToInsert">values of inserted row of data</param>
/// <param name="maskOfRow">bitmask of inserted row</param>
/// <returns>block index and index in block where row should be inserted</returns>
std::tuple<int, int> Table::GetIndex(std::vector<std::any> rowToInsert, std::vector<uint8_t> maskOfRow)
{
    int index = 0;
    CompareResult compareResult = CompareResult::Equal;

    int left = 0;
    int right = GetDataRangeInSortingColumn();

    if (right == 0)
    {
        return std::make_tuple(0, 0);
    }

    right -= 1;
    while (left <= right)
    {
        index = (left + right) / 2;

        compareResult = CompareRows(rowToInsert, maskOfRow, index);

        if (compareResult == CompareResult::Greater)
        {
            left = index + 1;
        }

        else if (compareResult == CompareResult::Lower)
        {
            right = index - 1;
        }

        else
            return GetIndicesFromTotalIndex(index, false);
    }

    return (compareResult == CompareResult::Greater ? GetIndicesFromTotalIndex(index + 1, false) :
                                                      GetIndicesFromTotalIndex(index, false));
}
#endif

const std::shared_ptr<Database>& Table::GetDatabase() const
{
    return database_;
}

const std::string& Table::GetName() const
{
    return name_;
}

void Table::SetTableName(const std::string& newTableName)
{
    name_ = newTableName;
}

int Table::GetBlockSize() const
{
    return blockSize_;
}

void Table::SetBlockSize(int32_t blockSize)
{
    blockSize_ = blockSize;
}

int32_t Table::GetBlockCount() const
{
    int32_t blockCount = 0;

    /* If there is at least one column in a table, return the block count of this column,
       because each column should have the same block counts */
    if (!columns_.empty())
    {
        blockCount = columns_.begin()->second.get()->GetBlockCount();
    }

    return blockCount;
}

int64_t Table::GetSize() const
{
    int64_t size = 0;
    for (auto& column : columns_)
    {
        if (column.second->GetSize() > size)
        {
            size = column.second->GetSize();
        }
    }
    return size;
}

const std::unordered_map<std::string, std::unique_ptr<IColumn>>& Table::GetColumns() const
{
    return columns_;
}

const std::vector<std::string>& Table::GetSortingColumns()
{
    return sortingColumns_;
}

void Table::SetSortingColumns(std::vector<std::string> columns)
{
    sortingColumns_ = columns;
}

void Table::AddSortingColumn(const std::string& sortingColumn)
{
    if (std::find(sortingColumns_.begin(), sortingColumns_.end(), sortingColumn) == sortingColumns_.end())
    {
        sortingColumns_.push_back(sortingColumn);
    }
    else
    {
        BOOST_LOG_TRIVIAL(debug) << "Attempt to insert duplicate sorting column.";
    }
}

void Table::RemoveSortingColumn(const std::string& sortingColumn)
{
    if (std::find(sortingColumns_.begin(), sortingColumns_.end(), sortingColumn) != sortingColumns_.end())
    {
        sortingColumns_.erase(std::find(sortingColumns_.begin(), sortingColumns_.end(), sortingColumn));
    }
    else
    {
        BOOST_LOG_TRIVIAL(debug) << "Attempt to erase non-existing sorting column.";
    }
}

void Table::RenameColumn(std::string oldColumnName, std::string newColumnName)
{
    std::unique_lock<std::mutex> lock(*columnsMutex_);
    columns_.at(oldColumnName)->SetColumnName(newColumnName);
    auto handler = columns_.extract(oldColumnName);
    handler.key() = newColumnName;
    columns_.insert(std::move(handler));
}

/// <summary>
/// Insert null values into new column which was added using alter table add column comand to match amount of data in blocks in columns which are already in database
/// </summary>
/// <param name="newColumnname">name of new column which was added using alter table add column</param>
void Table::InsertNullDataIntoNewColumn(std::string newColumnName)
{
    std::unique_lock<std::mutex> lock(*columnsMutex_);
    auto iterator = columns_.begin();

    if (iterator->second->GetName() == newColumnName)
    {
        iterator++;
    }

    auto& column = iterator->second;
    DataType type = column->GetColumnType();

    std::vector<int32_t> blocksSizes;

    switch (type)
    {
    case COLUMN_INT:
    {
        auto blocks = dynamic_cast<ColumnBase<int32_t>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_LONG:
    {
        auto blocks = dynamic_cast<ColumnBase<int64_t>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_DOUBLE:
    {
        auto blocks = dynamic_cast<ColumnBase<double>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_FLOAT:
    {
        auto blocks = dynamic_cast<ColumnBase<float>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_POINT:
    {
        auto blocks = dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_POLYGON:
    {
        auto blocks = dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_STRING:
    {
        auto blocks = dynamic_cast<ColumnBase<std::string>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    case COLUMN_INT8_T:
    {
        auto blocks = dynamic_cast<ColumnBase<int8_t>*>(column.get())->GetBlocksList();

        for (int32_t i = 0; i < blocks.size(); i++)
        {
            blocksSizes.push_back(blocks[i]->GetSize());
        }
    }
    break;

    default:
        throw std::runtime_error("Unsupported column type.");
        break;
    }


    auto newColumn = columns_.at(newColumnName).get();
    DataType newType = newColumn->GetColumnType();

    switch (newType)
    {
    case COLUMN_INT:
    {
        auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<int32_t>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_LONG:
    {
        auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<int64_t>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_DOUBLE:
    {
        auto castedColumn = dynamic_cast<ColumnBase<double>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<double>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_FLOAT:
    {
        auto castedColumn = dynamic_cast<ColumnBase<float>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<float>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_POINT:
    {
        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block =
                castedColumn->AddBlock(ColumnBase<QikkDB::Types::Point>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_POLYGON:
    {
        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(
                ColumnBase<QikkDB::Types::ComplexPolygon>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_STRING:
    {
        auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<std::string>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    case COLUMN_INT8_T:
    {
        auto castedColumn = dynamic_cast<ColumnBase<int8_t>*>(newColumn);
        for (int32_t i = 0; i < blocksSizes.size(); i++)
        {
            auto& block = castedColumn->AddBlock(ColumnBase<int8_t>::NullArray(blocksSizes[i]));

            for (int32_t j = 0; j < blocksSizes[i]; j++)
            {
                NullValues::SetBitInBitMask(block.GetNullBitmask(), j, static_cast<nullmask_t>(1U));
            }
        }
    }
    break;

    default:
        throw std::runtime_error("Unsupported column type.");
        break;
    }
}

void Table::AddConstraint(const std::string& constraintName,
                          ConstraintType constraintType,
                          std::vector<std::string> constraintColumns)
{
    std::vector<std::string> processedColumns;

    try
    {
        for (auto& constraintColumn : constraintColumns)
        {
            switch (constraintType)
            {
            case ConstraintType::CONSTRAINT_INDEX:
                AddSortingColumn(constraintColumn);
                break;
            case ConstraintType::CONSTRAINT_NOT_NULL:
                GetColumns().at(constraintColumn)->SetIsNullable(false);
                break;
            case ConstraintType::CONSTRAINT_UNIQUE:
                GetColumns().at(constraintColumn)->SetIsUnique(true);
                break;
            case ConstraintType::CONSTRAINT_NONE:
            default:
                break;
            }
            processedColumns.push_back(constraintColumn);
        }
        constraints_.insert({constraintName, {constraintType, constraintColumns}});
    }
    catch (std::exception& e)
    {
        for (auto& processedColumn : processedColumns)
        {
            switch (constraintType)
            {
            case ConstraintType::CONSTRAINT_INDEX:
                RemoveSortingColumn(processedColumn);
                break;
            case ConstraintType::CONSTRAINT_NOT_NULL:
                GetColumns().at(processedColumn)->SetIsNullable(true);
                break;
            case ConstraintType::CONSTRAINT_UNIQUE:
                GetColumns().at(processedColumn)->SetIsUnique(false);
                break;
            case ConstraintType::CONSTRAINT_NONE:
            default:
                break;
            }
        }
        throw;
    }
}

void Table::DropConstraint(const std::string& constraintName)
{
    auto& constraint = constraints_.at(constraintName);
    std::vector<std::string> processedColumns;

    try
    {
        for (auto& constraintColumn : constraint.second)
        {
            switch (constraint.first)
            {
            case ConstraintType::CONSTRAINT_INDEX:
                RemoveSortingColumn(constraintColumn);
                break;
            case ConstraintType::CONSTRAINT_NOT_NULL:
                GetColumns().at(constraintColumn)->SetIsNullable(true);
                break;
            case ConstraintType::CONSTRAINT_UNIQUE:
                GetColumns().at(constraintColumn)->SetIsUnique(false);
                break;
            case ConstraintType::CONSTRAINT_NONE:
            default:
                break;
            }
            processedColumns.push_back(constraintColumn);
        }
        constraints_.erase(constraintName);
    }
    catch (std::exception& e)
    {
        for (auto& processedColumn : processedColumns)
        {
            switch (constraint.first)
            {
            case ConstraintType::CONSTRAINT_INDEX:
                RemoveSortingColumn(processedColumn);
                break;
            case ConstraintType::CONSTRAINT_NOT_NULL:
                GetColumns().at(processedColumn)->SetIsNullable(false);
                break;
            case ConstraintType::CONSTRAINT_UNIQUE:
                GetColumns().at(processedColumn)->SetIsUnique(true);
                break;
            case ConstraintType::CONSTRAINT_NONE:
            default:
                break;
            }
        }
        throw;
    }
}

const std::unordered_map<std::string, std::pair<ConstraintType, std::vector<std::string>>>&
Table::GetConstraints() const
{
    return constraints_;
}

std::unordered_set<ConstraintType> Table::GetConstraintsForColumn(const std::string& columnName)
{
    std::unordered_set<ConstraintType> columnConstraints;

    if (std::find(sortingColumns_.begin(), sortingColumns_.end(), columnName) != sortingColumns_.end())
    {
        columnConstraints.insert(ConstraintType::CONSTRAINT_INDEX);
    }
    if (GetColumns().at(columnName)->GetIsNullable() == false)
    {
        columnConstraints.insert(ConstraintType::CONSTRAINT_NOT_NULL);
    }
    if (GetColumns().at(columnName)->GetIsUnique())
    {
        columnConstraints.insert(ConstraintType::CONSTRAINT_UNIQUE);
    }
    return columnConstraints;
}

int32_t Table::GetSaveInterval() const
{
    return saveInterval_;
}

void Table::SetSaveInterval(const int32_t newSaveInterval)
{
    saveInterval_ = newSaveInterval;
}

/// <summary>
/// Removes column from columns.
/// </summary>
/// <param name="columnName">Name of column to be removed.</param>
void Table::EraseColumn(const std::string& columnName)
{
    std::unique_lock<std::mutex> lock(*columnsMutex_);
    columns_.erase(columnName);
}

/// <summary>
/// Initializes a new instance of the class. Also gets from database
/// the block size and initializes with this value the private variable blockSize. Finally, it initializes columnsMutex_.
/// </summary>
/// <param name="database">Pointer to the database which will contains the new table.</param>
/// <param name="name">Name of the newly created table.</param>
/// <param name="blockSize">Table block size. If not specified, as the default value a database block size will be used.</param>
Table::Table(const std::shared_ptr<Database>& database, const char* name, const int32_t blockSize)
: database_(database), name_(name), columnsMutex_(std::make_unique<std::mutex>())
{
    if (blockSize == -1)
    {
        // if table block size was not specified, use as the default value the block size from database
        blockSize_ = database->GetBlockSize();
    }
    else
    {
        // if table block size was specified, use it as table block size for this particular table
        blockSize_ = blockSize;
    }

    BOOST_LOG_TRIVIAL(info) << "New table was created with block size: " << blockSize
                            << " and table name: " << name << ".";
}

/// <summary>
/// Insert new column with proper data type into the table.
/// </summary>
/// <param name="columnName">Name of column.</param>
/// <param name="dataType">Data type of colum.n</param>
void Table::CreateColumn(const char* columnName, DataType columnType, bool isNullable, bool isUnique)
{
    Context::getInstance().CheckColumnsLimit(columns_.size() - 1);

    std::unique_ptr<IColumn> column;

    if (columnType == COLUMN_INT)
    {
        column = std::make_unique<ColumnBase<int32_t>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_LONG)
    {
        column = std::make_unique<ColumnBase<int64_t>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_DOUBLE)
    {
        column = std::make_unique<ColumnBase<double>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_FLOAT)
    {
        column = std::make_unique<ColumnBase<float>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_STRING)
    {
        column = std::make_unique<ColumnBase<std::string>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_POLYGON)
    {
        column = std::make_unique<ColumnBase<QikkDB::Types::ComplexPolygon>>(columnName, blockSize_,
                                                                             isNullable, isUnique);
    }
    else if (columnType == COLUMN_POINT)
    {
        column = std::make_unique<ColumnBase<QikkDB::Types::Point>>(columnName, blockSize_, isNullable, isUnique);
    }
    else if (columnType == COLUMN_INT8_T)
    {
        column = std::make_unique<ColumnBase<int8_t>>(columnName, blockSize_, isNullable, isUnique);
    }
    std::unique_lock<std::mutex> lock(*columnsMutex_);
    columns_.insert(std::make_pair(columnName, std::move(column)));
}

#ifndef __CUDACC__
/// <summary>
/// Insert data into proper column of table considering empty space of last block and maximum size
/// of blocks. Considering indices Considering isUnique constraint
/// </summary>
/// <param name="data">Name of column with inserting data.</param>
/// <param name="compress">Whether data will be compressed.</param>
void Table::InsertData(const std::unordered_map<std::string, std::any>& data,
                       bool compress,
                       const std::unordered_map<std::string, std::vector<nullmask_t>>& nullMasks)
{
    int oneColumnDataSize = GetDataSizeOfInsertedColumns(data);

    if (GetHasNotNullConstraints())
    {
        CheckNullableConstraintInData(nullMasks, oneColumnDataSize);
    }

    // This method check UNIQUE constraints and data which should be inserted if fit this requirement
    if (GetHasUniqueConstraints())
    {
        CheckUniqueConstraintInData(data, oneColumnDataSize);
    }

    // This block of code insert data according to set indices
    if (!sortingColumns_.empty())
    {
        std::vector<std::any> rowToInsert;
        std::vector<uint8_t> maskOfRow;
        std::vector<std::string> columnToIterate;

        int blockIndex;
        int indexInBlock;

        for (int i = 0; i < oneColumnDataSize; i++)
        {
            std::tie(rowToInsert, maskOfRow) = GetRowAndBitmaskOfInsertedData(data, i, nullMasks);
            std::tie(blockIndex, indexInBlock) = GetIndex(rowToInsert, maskOfRow);

            InsertValuesOnSpecificPosition(data, blockIndex, indexInBlock, i, nullMasks);
        }
    }
    else
    {
        for (const auto& column : columns_)
        {
            std::string columnName = column.first;
            if (data.find(columnName) != data.end())
            {
                const auto& wrappedData = data.at(columnName);
                if (nullMasks.find(columnName) != nullMasks.end())
                {
                    if (wrappedData.type() == typeid(std::vector<int32_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int32_t>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int32_t>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<int64_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int64_t>*>(columns_.find(columnName)->second.get());

                        castedColumn->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int64_t>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<double>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<double>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<double>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<double>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<float>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<float>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<float>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<float>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<std::string>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<std::string>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<std::string>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<std::string>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::ComplexPolygon>))
                    {
                        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(
                            columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(
                                std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::Point>))
                    {
                        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(
                            columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(
                                std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<int8_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int8_t>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<int8_t>>(wrappedData),
                                                 nullMasks.at(columnName), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int8_t>>(wrappedData));
                        }
                    }
                }
                else
                {
                    if (wrappedData.type() == typeid(std::vector<int32_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int32_t>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int32_t>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<int64_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int64_t>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int64_t>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<double>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<double>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<double>>(wrappedData), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<double>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<float>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<float>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<float>>(wrappedData), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<float>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<std::string>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<std::string>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<std::string>>(wrappedData),
                                                 -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<std::string>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::ComplexPolygon>))
                    {
                        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::ComplexPolygon>*>(
                            columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedData),
                                                 -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(
                                std::any_cast<std::vector<QikkDB::Types::ComplexPolygon>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<QikkDB::Types::Point>))
                    {
                        auto castedColumn = dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(
                            columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedData),
                                                 -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(
                                std::any_cast<std::vector<QikkDB::Types::Point>>(wrappedData));
                        }
                    }
                    else if (wrappedData.type() == typeid(std::vector<int8_t>))
                    {
                        auto castedColumn =
                            dynamic_cast<ColumnBase<int8_t>*>(columns_.find(columnName)->second.get());
                        castedColumn->InsertData(std::any_cast<std::vector<int8_t>>(wrappedData), -1, compress);
                        if (castedColumn->GetIsUnique())
                        {
                            castedColumn->InsertIntoHashmap(std::any_cast<std::vector<int8_t>>(wrappedData));
                        }
                    }
                }
            }
        }
    }
}
#endif

/// <summary>
/// Search for column according to its name.
/// </summary>
/// <param name="column">Name of column.</param>
/// <returns>Return true, if table contains particular column. Returns false, if table does not contains particular column.</returns>
bool Table::ContainsColumn(const char* column)
{
    auto search = columns_.find(column);
    if (search != columns_.end())
    {
        return true;
    }
    return false;
}

/// <summary>
/// Find out the index of binary index group of blocks, to which the row data will be inserted.
/// The main question that decides the group is as follows: 'Is value > average?'.
/// </summary>
/// <param name="rowData">Row of data from .CSV file, that is inserting into the table.</param>
/// <param name="columns">Columns in the same order as fields in row of data.</param>
/// <returns>Index of binary index group of blocks.</returns>
int32_t Table::AssignGroupId(std::vector<std::any>& rowData, std::vector<std::unique_ptr<IColumn>>& columns)
{
    int32_t index = 0;

    for (int32_t i = 0; i < columns.size(); i++)
    {
        bool b = false;

        // if initial average is not set, assign all values in the default group:
        if (!columns[i]->GetInitAvgIsSet())
        {
            return -1;
        }

        switch (columns[i]->GetColumnType())
        {
        case COLUMN_INT:
            if (std::any_cast<int32_t>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_LONG:
            if (std::any_cast<int64_t>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_FLOAT:
            if (std::any_cast<float>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_DOUBLE:
            if (std::any_cast<double>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_POINT:
            if (std::any_cast<float>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_POLYGON:
            if (std::any_cast<float>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_STRING:
            if (std::any_cast<float>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        case COLUMN_INT8_T:
            if (std::any_cast<int8_t>(rowData[i]) > columns[i]->GetInitAvg())
            {
                b = true;
            }
            break;
        default:
            throw std::domain_error(
                "Unsupported data type (when importing database from CSV file): " +
                std::to_string(columns[i]->GetColumnType()));
        }

        index += 2 * i + b;
    }

    return index;
}

/// <summary>
/// Find Ids for all groups needed for binary index per particular table. Do not include -1 as index, because it is default group id.
/// </summary>
/// <param name="columns">Columns of a table.</param>
/// <returns>Ids of all groups for binary index.</returns>
std::vector<int32_t> Table::GetTableGroupIds(std::unordered_map<std::string, std::unique_ptr<IColumn>>& columns)
{
    std::vector<int32_t> groupIds;

    for (int32_t i = 0; i < columns.size(); i++)
    {
        groupIds.push_back(i);
    }

    return groupIds;
}

/// <summary>
/// Find Ids for all groups needed for binary index per particular table. Do not include -1 as index, because it is default group id.
/// </summary>
/// <param name="columns">Columns of a table.</param>
/// <returns>Ids of all groups for binary index.</returns>
std::vector<int32_t> Table::GetTableGroupIds(std::vector<std::unique_ptr<IColumn>>& columns)
{
    std::vector<int32_t> groupIds;

    for (int32_t i = 0; i < columns.size(); i++)
    {
        groupIds.push_back(i);
    }

    return groupIds;
}
