#include "Table.h"

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

Table::Table(const std::shared_ptr<Database> database, std::string name)
{
}

void Table::createColumn(std::string columnName, ColumnType columnType)
{
}

bool Table::containsColumn(std::string)
{
	return false;
}
