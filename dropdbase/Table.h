#pragma once
#include "Database.h"
#include "IColumn.h"
#include "DataType.h"
#include <memory>
#include <string>
#include <unordered_map>
#ifndef __CUDACC__
#include <any>
#endif

class Table
{
private:
	const std::shared_ptr<Database> database;
	std::string name;
	int blockSize;
	std::unordered_map<std::string, std::unique_ptr<IColumn>> columns;

public:
	const std::shared_ptr<Database> &GetDatabase();
	const std::string &GetName();
	int GetBlockSize();
	const std::unordered_map<std::string, std::unique_ptr<IColumn>> &GetColumns() const;

	Table(const std::shared_ptr<Database> &database, const char* name);
	void CreateColumn(const char* columnName, DataType columnType);
#ifndef __CUDACC__
	void InsertData(const std::unordered_map<std::string, std::any> &data);
#endif
	bool ContainsColumn(const char* column);
};
