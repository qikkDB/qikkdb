#pragma once
#include "Database.h"
#include "IColumn.h"
#include "ColumnType.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <any>

class Table
{
private:
	const std::shared_ptr<Database> database;
	std::string name;
	int blockSize;
	std::unordered_map<std::string, IColumn> columns;

public:
	const std::shared_ptr<Database> &getDatabase();
	const std::string &getName();
	int getBlockSize();
	const std::unordered_map<std::string, IColumn> &getColumns();

	Table(const std::shared_ptr<Database> database, std::string name);
	void createColumn(std::string columnName, ColumnType columnType);
	void insertData(const std::unordered_map<std::string, std::any> &data);
	bool containsColumn(std::string column);
};