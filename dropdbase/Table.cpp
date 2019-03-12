#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include <cstdint>

const std::shared_ptr<Database>& Table::GetDatabase() const
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

const std::unordered_map<std::string, std::unique_ptr<IColumn>>& Table::GetColumns() const
{
	return columns;
}

Table::Table(const std::shared_ptr<Database> &database, const char* name) : database(database), name(name), columnsMutex_(std::make_unique<std::mutex>())
{
	blockSize = database->GetBlockSize();
}

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
void Table::InsertData(const std::unordered_map<std::string, std::any>& data)
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
#endif

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
/// <param name="allColumns">Columns in the same order as fields in row of data.</param>
/// <param name="indexColumns">Names of columns on which binary index will be created in order from most significant to least significant.</param>
/// <returns>Index of binary index group of blocks.</returns>
int32_t Table::AssignGroupId(std::vector<std::any>& rowData, std::unordered_map<std::string, std::unique_ptr<IColumn>>& allColumns, std::vector<std::string>& indexColumns)
{
	std::vector<std::string> tempIndexColumns;

	//find out if there is at least one column with set initAvg_ so it can make index:
	for (std::string indexColumn : indexColumns)
	{
		if (allColumns[indexColumn]->GetInitAvgIsSet())
		{
			tempIndexColumns.push_back(indexColumn);
		}
	}

	//if there is no column according to which it make sense to create index:
	if (tempIndexColumns.size() == 0)
	{
		return -1;
	}

	int32_t index = 0;

	for (std::string tempIndexColumn : tempIndexColumns)
	{
		bool b = false;

		//TODO i musi byt spojene s allColumns poradim
		if (rowData[i] > allColumns[tempIndexColumn]->GetInitAvg())
		{
			b = true;
		}

		index += 2 * i + b;
	}

	return index;
}
