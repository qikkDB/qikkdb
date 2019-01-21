#include "Table.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"

const std::shared_ptr<Database>& Table::GetDatabase()
{
	return database;
}

const std::string & Table::GetName()
{
	return name;
}

int Table::GetBlockSize()
{
	return blockSize;
}

const std::unordered_map<std::string, std::unique_ptr<IColumn>>& Table::GetColumns()
{
	return columns;
}

Table::Table(const std::shared_ptr<Database> database, std::string name) : database(database), name(name)
{
	blockSize = database->GetBlockSize();
}

void Table::CreateColumn(const std::string &columnName, DataType columnType)
{
	std::unique_ptr<IColumn> column;

	if (columnType == COLUMN_INT)
	{
		column = std::make_unique<ColumnBase<int>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_LONG)
	{
		column = std::make_unique<ColumnBase<long>>(columnName, blockSize);
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
		column = std::make_unique<ColumnBase<ColmnarDB::Types::Polygon>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_POINT)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::Point>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_BOOL)
	{
		column = std::make_unique<ColumnBase<bool>>(columnName, blockSize);
	}
	columns.insert(std::make_pair(columnName, std::move(column)));
}

void Table::InsertData(const std::unordered_map<std::string, std::any>& data)
{
	for (const auto& column : columns)
	{
		std::string columnName = column.first;
		auto search = data.find(columnName);
		if (search != data.end())
		{
			const auto &wrappedData = data.at(columnName);
			if (wrappedData.type() == typeid(std::vector<int>))
			{
				dynamic_cast<ColumnBase<int>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<long>))
			{
				dynamic_cast<ColumnBase<long>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<long>>(wrappedData));
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
			else if (wrappedData.type() == typeid(std::vector<bool>))
			{
				dynamic_cast<ColumnBase<bool>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<bool>>(wrappedData));
			}
		}
	}
}

bool Table::ContainsColumn(std::string column)
{
	auto search = columns.find(column);
	if (search != columns.end()) {
		return true;
	}
	return false;
}
