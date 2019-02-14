#pragma once
#include "IColumn.h"
#include "DataType.h"
#include <memory>
#include <string>
#include <unordered_map>
#ifndef __CUDACC__
#include <any>
#endif

class Database;

class Table
{
private:
	const std::shared_ptr<Database> database;
	std::string name;
	int32_t blockSize;
	std::unordered_map<std::string, std::unique_ptr<IColumn>> columns;
	std::string sortingColumn = "";
#ifndef __CUDACC__
	void InsertValuesInNonIndexColumns(const std::unordered_map<std::string, std::any> &data, int indexBlock, int indexInBlock, std::string sortingColumn, int iterator);
#endif

public:
	const std::shared_ptr<Database> &GetDatabase();
	const std::string &GetName();
	int32_t GetBlockSize();
	int32_t GetBlockCount();
	const std::unordered_map<std::string, std::unique_ptr<IColumn>> &GetColumns() const;
	std::string GetSortingColumn();
	void SetSortingColumn(std::string column);

	Table(const std::shared_ptr<Database> &database, const char* name);
	void CreateColumn(const char* columnName, DataType columnType);
#ifndef __CUDACC__
	void InsertData(const std::unordered_map<std::string, std::any> &data);
#endif
	bool ContainsColumn(const char* column);
};
