#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include <cstdint>


#ifndef __CUDACC__
void Table::InsertValuesOnSpecificPosition(const std::unordered_map<std::string, std::any>& data, int indexBlock, int indexInBlock, int iterator)
{
    for (const auto& column : columns)
    {
        const std::string columnName = column.first;
        auto search = data.find(columnName);
        if (search != data.end())
        {
            auto currentColumn = (columns.find(columnName)->second.get());
            const auto &wrappedData = data.at(columnName);

            if (wrappedData.type() == typeid(std::vector<int32_t>))
            {
                std::vector<int32_t> dataInt = std::any_cast<std::vector<int32_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataInt[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<int64_t>))
            {
                std::vector<int64_t> dataLong = std::any_cast<std::vector<int64_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataLong[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<double>))
            {
                std::vector<double> dataDouble = std::any_cast<std::vector<double>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataDouble[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<float>))
            {
                std::vector<float> dataFloat = std::any_cast<std::vector<float>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataFloat[iterator]);
            }
			//TODO string, point, polygon
        }
    }
}

int32_t Table::getDataSizeOfInsertedColumns(const std::unordered_map<std::string, std::any>& data)
{
    int size;

    auto firstSortingColumn = (columns.find(sortingColumns[0])->second.get());
    const auto& dataOfFirstColumn = data.at(sortingColumns[0]);

    if (dataOfFirstColumn.type() == typeid(std::vector<int32_t>))
    {
        std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
	}

	if (dataOfFirstColumn.type() == typeid(std::vector<int64_t>))
    {
        std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	if (dataOfFirstColumn.type() == typeid(std::vector<double>))
    {
        std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	if (dataOfFirstColumn.type() == typeid(std::vector<float>))
    {
        std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	//TODO for polygon, point and string

		return size;
}

int32_t Table::getDataRangeInSortingColumn()
{
	int size = 0;

	auto firstSortingColumn = (columns.find(sortingColumns[0])->second.get());
	auto columnType = firstSortingColumn->GetColumnType();

	if (columnType == COLUMN_INT)
	{
		auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	if (columnType == COLUMN_LONG)
	{
		auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	if (columnType == COLUMN_DOUBLE)
	{
		auto castedColumn = dynamic_cast<ColumnBase<double>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	if (columnType == COLUMN_FLOAT)
	{
		auto castedColumn = dynamic_cast<ColumnBase<float>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	return size;
}
#endif

const std::shared_ptr<Database>& Table::GetDatabase()
{
	return database;
}

const std::string & Table::GetName() const
{
	return name;
}

int Table::GetBlockSize() const
{
	return blockSize;
}

int32_t Table::GetBlockCount() const
{
	for (auto& column : columns) 
	{
		return column.second.get()->GetBlockCount();
	}
	return 0;
}

int64_t Table::GetSize() const
{
	int64_t size = 0;
	for (auto& column : columns)
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
	return columns;
}

std::vector<std::string> Table::GetSortingColumns()
{
	return sortingColumns;
}

void Table::SetSortingColumns(std::vector<std::string> columns)
{
	sortingColumns = columns;
}

/// <summary>
/// Removes column from columns.
/// </summary>
/// <param name="columnName">Name of column to be removed.</param>
void Table::EraseColumn(std::string & columnName)
{
	columns.erase(columnName);
}

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.Table"/> class. Also gets from database
/// the block size and initializes with this value the private variable blockSize. Finally, it initializes columnsMutex_.
/// </summary>
/// <param name="database">Pointer to the database which will contains the new table.</param>
/// <param name="name">Name of the newly created table.</param>
Table::Table(const std::shared_ptr<Database> &database, const char* name) : database(database), name(name), columnsMutex_(std::make_unique<std::mutex>())
{
	blockSize = database->GetBlockSize();
}

/// <summary>
/// Insert new column with proper data type into the table.
/// </summary>
/// <param name="columnName">Name of column.</param>
/// <param name="dataType">Data type of colum.n</param>
void Table::CreateColumn(const char* columnName, DataType columnType)
{
	std::unique_ptr<IColumn> column;

	if (columnType == COLUMN_INT)
	{
		column = std::make_unique<ColumnBase<int32_t>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_LONG)
	{
		column = std::make_unique<ColumnBase<int64_t>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_DOUBLE)
	{
		column = std::make_unique<ColumnBase<double>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_FLOAT)
	{
		column = std::make_unique<ColumnBase<float>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_STRING)
	{
		column = std::make_unique<ColumnBase<std::string>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_POLYGON)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::ComplexPolygon>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_POINT)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::Point>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_INT8_T)
	{
		column = std::make_unique<ColumnBase<int8_t>>(columnName, blockSize);
	}
	std::unique_lock<std::mutex> lock(*columnsMutex_);
	columns.insert(std::make_pair(columnName, std::move(column)));
}

#ifndef __CUDACC__
/// <summary>
/// Insert data into proper column of table considering empty space of last block and maximum size of blocks.
/// </summary>
/// <param name="data">Name of column with inserting data.</param>
void Table::InsertData(const std::unordered_map<std::string, std::any>& data)
{
	if (!sortingColumns.empty())
	{
        int oneColumnDataSize = getDataSizeOfInsertedColumns(data);
		
		for (int i = 0; i < oneColumnDataSize; i++)
        {
			int range = getDataRangeInSortingColumn();
            //int range = INT_MAX;
            int blockIndex = 0;
            int indexInBlock = 0;

            for (auto sortingColumn : sortingColumns)
            {
                auto currentSortingColumn = (columns.find(sortingColumn)->second.get());
                const auto& wrappedCurrentSortingColumnData = data.at(sortingColumn);

                if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int32_t>))
                {
                    std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);

					
					std::tie(blockIndex, indexInBlock, range) = castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int64_t>))
                {
                    std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<float>))
                {
                    std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<double>))
                {
                    std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				//TODO string, polygon, point
            }

			InsertValuesOnSpecificPosition(data,blockIndex,indexInBlock,i);
			
		}
	}

	else
	{
		for (const auto& column : columns)
		{
			std::string columnName = column.first;
			auto search = data.find(columnName);
			if (search != data.end())
			{
				const auto &wrappedData = data.at(columnName);
				if (wrappedData.type() == typeid(std::vector<int32_t>))
				{
					dynamic_cast<ColumnBase<int32_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<int64_t>))
				{
					dynamic_cast<ColumnBase<int64_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<double>))
				{
					dynamic_cast<ColumnBase<double>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<double>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<float>))
				{
					dynamic_cast<ColumnBase<float>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<float>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<std::string>))
				{
					dynamic_cast<ColumnBase<std::string>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<std::string>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
				{
					dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
				{
					dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData));
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
	auto search = columns.find(column);
	if (search != columns.end()) {
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

		//if initial average is not set, assign all values in the default group:
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
			throw std::domain_error("Unsupported data type (when importing database from CSV file).");
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
