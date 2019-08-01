#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include <cstdint>


#ifndef __CUDACC__
/// <summary>
/// Insert row of data to database on specific position
/// </summary>
/// <param name="data">column name with inserting data</param>
/// <param name="indexBlock">index of block where data will be inserted</param>
/// <param name="indexInBlock">index in block where data will be inserted</param>
/// <param name="iterator">index of row of data</param>
/// <param name="nullMask">column name with bitmask</param>
void Table::InsertValuesOnSpecificPosition(const std::unordered_map<std::string, std::any>& data, int indexBlock, int indexInBlock, int iterator, const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks)
{
    for (const auto& column : columns)
    {
        const std::string columnName = column.first;
        auto search = data.find(columnName);
        if (search != data.end())
        {
			int8_t isNullValue = false;
			int bitMaskIdx = (iterator / (sizeof(char) * 8));
			int shiftIdx = (iterator % (sizeof(char) * 8));
			if (nullMasks.find(columnName) != nullMasks.end())
			{
				isNullValue = (nullMasks.at(columnName)[bitMaskIdx] >> shiftIdx) & 1;
			}


            auto currentColumn = (columns.find(columnName)->second.get());
            const auto &wrappedData = data.at(columnName);

            if (wrappedData.type() == typeid(std::vector<int32_t>))
            {
                std::vector<int32_t> dataInt = std::any_cast<std::vector<int32_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataInt[iterator], -1, isNullValue);
            }
            else if (wrappedData.type() == typeid(std::vector<int64_t>))
            {
                std::vector<int64_t> dataLong = std::any_cast<std::vector<int64_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataLong[iterator], -1, isNullValue);
            }
            else if (wrappedData.type() == typeid(std::vector<double>))
            {
                std::vector<double> dataDouble = std::any_cast<std::vector<double>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataDouble[iterator], -1, isNullValue);
            }
            else if (wrappedData.type() == typeid(std::vector<float>))
            {
                std::vector<float> dataFloat = std::any_cast<std::vector<float>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataFloat[iterator], -1, isNullValue);
            }
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
			{
				std::vector<ColmnarDB::Types::Point> dataPoint = std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData);
				auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(currentColumn);
				castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataPoint[iterator]);
			}
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
			{
				std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon = std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData);
				auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(currentColumn);
				castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataPolygon[iterator]);
			}
			else if (wrappedData.type() == typeid(std::vector<std::string>))
			{
				std::vector<std::string> dataString = std::any_cast<std::vector<std::string>>(wrappedData);
				auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(currentColumn);
				castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataString[iterator]);
			}
        }
    }
}

/// <summary>
/// Gets count ofrows of inserted  data
/// </summary>
/// <param name="data">unordered map of columnName and data that should be inserted in this column</param>
/// <returns>Count of rows of inserted data</returns>
int32_t Table::getDataSizeOfInsertedColumns(const std::unordered_map<std::string, std::any>& data)
{
    int size = 0;

    const auto& dataOfFirstColumn = data.at(sortingColumns[0]);

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

	return size;
}

/// <summary>
/// Gets count of rows that are already inserted in database - information is getting as count of data in first sorting column
/// </summary>
/// <returns>count of rows of data in database</returns>
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

	else if (columnType == COLUMN_LONG)
	{
		auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	else if (columnType == COLUMN_DOUBLE)
	{
		auto castedColumn = dynamic_cast<ColumnBase<double>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
		int blockCount = castedColumn->GetBlockCount();

		for (int i = 0; i < blockCount; i++)
		{
			size += blocks[i]->GetSize();
		}
	}

	else if (columnType == COLUMN_FLOAT)
	{
		auto castedColumn = dynamic_cast<ColumnBase<float>*>(firstSortingColumn);
		auto &blocks = castedColumn->GetBlocksList();
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
/// <returns>tuple of row and bitmask of this row from speciific position</returns>
std::tuple<std::vector<std::any>, std::vector<int8_t>> Table::GetRowAndBitmaskOfInsertedData(const std::unordered_map<std::string, std::any>& data, int iterator, const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks)
{
	std::vector<std::any> resultRow;
	std::vector<int8_t> maskOfRow;

	for (auto sortingColumn : sortingColumns)
	{
		int8_t isNullValue = 0;
		int bitMaskIdx = (iterator / (sizeof(char) * 8));
		int shiftIdx = (iterator % (sizeof(char) * 8));
		if (nullMasks.find(sortingColumn) != nullMasks.end())
		{
			isNullValue = (nullMasks.at(sortingColumn)[bitMaskIdx] >> shiftIdx) & 1;
		}

		maskOfRow.push_back(isNullValue);

		const auto& wrappedCurrentSortingColumnData = data.at(sortingColumn);

		if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int32_t>))
		{
			std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(wrappedCurrentSortingColumnData);
			resultRow.push_back(dataIndexedColumn[iterator]);
		}

		else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int64_t>))
		{
			std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(wrappedCurrentSortingColumnData);
			resultRow.push_back(dataIndexedColumn[iterator]);
		}

		else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<float>))
		{
			std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(wrappedCurrentSortingColumnData);
			resultRow.push_back(dataIndexedColumn[iterator]);
		}

		else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<double>))
		{
			std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(wrappedCurrentSortingColumnData);
			resultRow.push_back(dataIndexedColumn[iterator]);
		}

		else if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<std::string>))
		{
			std::vector<std::string> dataIndexedColumn = std::any_cast<std::vector<std::string>>(wrappedCurrentSortingColumnData);
			resultRow.push_back(dataIndexedColumn[iterator]);
		}
	}

	return std::make_tuple(resultRow, maskOfRow);
}

/// <summary>
/// Gets block index and index in block from total index to database
/// </summary>
/// <param name="index">index to database</param>
/// <param name="positionToCompare">whether method is used to get indices to compare row from these indices or we're looking for indices to insert row</param>
/// <returns>block index, index in block</returns>
std::tuple<int, int> Table::GetIndicesFromTotalIndex(int index, bool positionToCompare)
{
	int32_t blockIndex = 0;
	int32_t indexInBlock = index;

	auto firstSortingColumn = (columns.find(sortingColumns[0])->second.get());
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
std::tuple<std::vector<std::any>, std::vector<int8_t>> Table::GetRowAndBitmaskOnIndex(int index)
{
	int blockIndex;
	int indexInBlock;

	std::tie(blockIndex, indexInBlock) = GetIndicesFromTotalIndex(index, true);
	std::vector<std::any> resultRow;
	std::vector<int8_t> maskOfRow;

	int8_t isNullValue = 0;
	int bitMaskIdx = (indexInBlock / (sizeof(char) * 8));
	int shiftIdx = (indexInBlock % (sizeof(char) * 8));

	for (auto sortingColumn : sortingColumns)
	{
		auto currentSortingColumn = (columns.find(sortingColumn)->second.get()); 
		auto columnType = currentSortingColumn->GetColumnType();

		if (columnType == COLUMN_INT)
		{
			auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);
			resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

			isNullValue = (castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask()[bitMaskIdx] >> shiftIdx) & 1;
			maskOfRow.push_back(isNullValue);
		}

		else if (columnType == COLUMN_LONG)
		{
			auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentSortingColumn);
			resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

			isNullValue = (castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask()[bitMaskIdx] >> shiftIdx) & 1;
			maskOfRow.push_back(isNullValue);
		}

		else if (columnType == COLUMN_FLOAT)
		{
			auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentSortingColumn);
			resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

			isNullValue = (castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask()[bitMaskIdx] >> shiftIdx) & 1;
			maskOfRow.push_back(isNullValue);
		}

		else if (columnType == COLUMN_DOUBLE)
		{
			auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentSortingColumn);
			resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);

			isNullValue = (castedColumn->GetBlocksList()[blockIndex]->GetNullBitmask()[bitMaskIdx] >> shiftIdx) & 1;
			maskOfRow.push_back(isNullValue);
		}

		else if (columnType == COLUMN_STRING)
		{
			auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(currentSortingColumn);
			resultRow.push_back(castedColumn->GetBlocksList()[blockIndex]->GetData()[indexInBlock]);


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
/// <returns>one of enum value - Greater, Lower, Equal - according of relationship of inserted row and row from database</returns>
Table::CompareResult Table::CompareRows(std::vector<std::any> rowToInsert, std::vector<int8_t> maskOfInsertRow, int index)
{
	std::vector<std::any> rowToCompare;
	std::vector<int8_t> maskOfCompareRow;

	std::tie(rowToCompare, maskOfCompareRow) = GetRowAndBitmaskOnIndex(index);

	for (int i = 0; i < sortingColumns.size(); i++)
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
			if ((columns.find(sortingColumns[i])->second.get())->GetColumnType() == COLUMN_INT)
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

			else if ((columns.find(sortingColumns[i])->second.get())->GetColumnType() == COLUMN_LONG)
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

			else if ((columns.find(sortingColumns[i])->second.get())->GetColumnType() == COLUMN_DOUBLE)
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

			else if ((columns.find(sortingColumns[i])->second.get())->GetColumnType() == COLUMN_FLOAT)
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

			else if ((columns.find(sortingColumns[i])->second.get())->GetColumnType() == COLUMN_STRING)
			{
				if (std::any_cast<std::string>(rowToInsert[i]) < std::any_cast<std::string>(rowToCompare[i]))
				{
					return CompareResult::Lower;
				}

				else if (std::any_cast<std::string>(rowToInsert[i]) > std::any_cast<std::string>(rowToCompare[i]))
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
std::tuple<int, int> Table::GetIndex(std::vector<std::any> rowToInsert, std::vector<int8_t> maskOfRow)
{
	int index;
	CompareResult compareResult;

	int left = 0;
	int right = getDataRangeInSortingColumn();

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

		else return GetIndicesFromTotalIndex(index, false);
	}

	return (compareResult == CompareResult::Greater ? GetIndicesFromTotalIndex(index + 1, false) : GetIndicesFromTotalIndex(index, false));
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
void Table::CreateColumn(const char* columnName, DataType columnType, bool isNullable)
{
	std::unique_ptr<IColumn> column;

	if (columnType == COLUMN_INT)
	{
		column = std::make_unique<ColumnBase<int32_t>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_LONG)
	{
		column = std::make_unique<ColumnBase<int64_t>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_DOUBLE)
	{
		column = std::make_unique<ColumnBase<double>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_FLOAT)
	{
		column = std::make_unique<ColumnBase<float>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_STRING)
	{
		column = std::make_unique<ColumnBase<std::string>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_POLYGON)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::ComplexPolygon>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_POINT)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::Point>>(columnName, blockSize, isNullable);
	}
	else if (columnType == COLUMN_INT8_T)
	{
		column = std::make_unique<ColumnBase<int8_t>>(columnName, blockSize, isNullable);
	}
	std::unique_lock<std::mutex> lock(*columnsMutex_);
	columns.insert(std::make_pair(columnName, std::move(column)));
}


#ifndef __CUDACC__
/// <summary>
/// Insert data into proper column of table considering empty space of last block and maximum size of blocks.
/// </summary>
/// <param name="data">Name of column with inserting data.</param>
/// <param name="compress">Whether data will be compressed.</param>
void Table::InsertData(const std::unordered_map<std::string, std::any>& data, bool compress, const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks)
{
	if (!sortingColumns.empty())
	{
		int oneColumnDataSize = getDataSizeOfInsertedColumns(data);
		int blockIndex;
		int indexInBlock;

		std::vector<std::any> rowToInsert;
		std::vector<int8_t> maskOfRow;


		for (int i = 0; i < oneColumnDataSize; i++)
		{
			std::tie(rowToInsert, maskOfRow) = GetRowAndBitmaskOfInsertedData(data, i, nullMasks);
			std::tie(blockIndex, indexInBlock) = GetIndex(rowToInsert, maskOfRow);

			InsertValuesOnSpecificPosition(data, blockIndex, indexInBlock, i, nullMasks);
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
				if(nullMasks.find(columnName) != nullMasks.end())
				{
					if (wrappedData.type() == typeid(std::vector<int32_t>))
					{
						dynamic_cast<ColumnBase<int32_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<int64_t>))
					{
						dynamic_cast<ColumnBase<int64_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<double>))
					{
						dynamic_cast<ColumnBase<double>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<double>>(wrappedData), nullMasks.at(columnName), -1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<float>))
					{
						dynamic_cast<ColumnBase<float>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<float>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<std::string>))
					{
						dynamic_cast<ColumnBase<std::string>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<std::string>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
					{
						dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
					{
						dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData), nullMasks.at(columnName),-1,compress);
					}							
				}
				else
				{
					if (wrappedData.type() == typeid(std::vector<int32_t>))
					{
						dynamic_cast<ColumnBase<int32_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<int64_t>))
					{
						dynamic_cast<ColumnBase<int64_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<double>))
					{
						dynamic_cast<ColumnBase<double>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<double>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<float>))
					{
						dynamic_cast<ColumnBase<float>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<float>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<std::string>))
					{
						dynamic_cast<ColumnBase<std::string>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<std::string>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
					{
						dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData),-1,compress);
					}
					else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
					{
						dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData),-1,compress);
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
			throw std::domain_error("Unsupported data type (when importing database from CSV file): " + std::to_string(columns[i]->GetColumnType()));
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
