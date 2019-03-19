#pragma once
#include "IColumn.h"
#include "DataType.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#ifndef __CUDACC__
#include <any>
#endif

class Database;

class Table
{
private:
	const std::shared_ptr<Database>& database;
	std::string name;
	int32_t blockSize;
	std::unordered_map<std::string, std::unique_ptr<IColumn>> columns;
	std::unique_ptr<std::mutex> columnsMutex_;

public:
	const std::shared_ptr<Database> &GetDatabase() const;
	const std::string &GetName() const;
	int32_t GetBlockSize() const;
	int32_t GetBlockCount() const;
	const std::unordered_map<std::string, std::unique_ptr<IColumn>> &GetColumns() const;

	Table(const std::shared_ptr<Database> &database, const char* name);
	void CreateColumn(const char* columnName, DataType columnType);
#ifndef __CUDACC__
	void InsertData(const std::unordered_map<std::string, std::any> &data);
#endif
	bool ContainsColumn(const char* column);
	int32_t AssignGroupId(std::vector<std::any>& rowData, std::vector<std::unique_ptr<IColumn>>& columns);
	std::vector<int32_t> GetTableGroupIds(std::unordered_map<std::string, std::unique_ptr<IColumn>>& columns);
	std::vector<int32_t> GetTableGroupIds(std::vector<std::unique_ptr<IColumn>>& columns);
};
