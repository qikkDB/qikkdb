#include "Table.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"

const std::shared_ptr<Database>& Table::getDatabase()
{
	return database;
}

const std::string & Table::getName()
{
	return name;
}

int Table::getBlockSize()
{
	return blockSize;
}

const std::unordered_map<std::string, IColumn>& Table::getColumns()
{
	return columns;
}

Table::Table(const std::shared_ptr<Database> database, std::string name) : database(database), name(name)
{
	//blockSize = database.GetBlockSize();
}

void Table::createColumn(std::string columnName, ColumnType columnType)
{
	IColumn column;

	if (columnType == INT)
	{
		ColumnBase<int> column(columnName, blockSize);
	}
	else if (columnType == LONG)
	{
		ColumnBase<long> column(columnName, blockSize);
	}
	else if (columnType == DOUBLE)
	{
		ColumnBase<double> column(columnName, blockSize);
	}
	else if (columnType == FLOAT)
	{
		ColumnBase<float> column(columnName, blockSize);
	}
	else if (columnType == STRING)
	{
		ColumnBase<std::string> column(columnName, blockSize);
	}
	else if (columnType == COMPLEXPOLYGON)
	{
		ColumnBase<ColmnarDB::Types::ComplexPolygon> column(columnName, blockSize);
	}
	else if (columnType == POINT)
	{
		ColumnBase<ColmnarDB::Types::Point> column(columnName, blockSize);
	}
	columns.insert(std::make_pair(columnName, column));
}

void Table::insertData(const std::unordered_map<std::string, std::any>& data)
{
	for(auto column : columns)
	{
		std::string columnName = column.first;
		auto search = data.find(columnName);
		if (search != data.end())
		{
			const auto &wrappedData = data.at(columnName);
			if (wrappedData.type() == typeid(std::vector<int>))
			{
				dynamic_cast<ColumnBase<int>*>(& columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<int>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<long>))
			{
				dynamic_cast<ColumnBase<long>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<long>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<double>))
			{
				dynamic_cast<ColumnBase<double>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<double>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<float>))
			{
				dynamic_cast<ColumnBase<float>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<float>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<std::string>))
			{
				dynamic_cast<ColumnBase<std::string>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<std::string>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
			{
				dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData));
			}
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
			{
				dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(&columns.find(columnName)->second)->InsertData(std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData));
			}
		}
	}
}

bool Table::containsColumn(std::string column)
{
	auto search = columns.find(column);
	if (search != columns.end()) {
		return true;
	}
	return false;
}
