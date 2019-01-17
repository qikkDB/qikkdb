#pragma once

#include <unordered_map>
#include <spdlog/spdlog.h>
#include <memory.h>

#include "Table.h"
#include "ColumnType.h"

/// <summary>
/// The main class representing database containing tables with data
/// </summary>
class Database
{
private:
	static std::unordered_map<std::string, std::shared_ptr<Database>> loadedDatabases_;
	std::string name_;
	int blockSize_;
	std::unordered_map<std::string, Table> tables_;

public:
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
	/// </summary>
	/// <param name="databaseName">Database name.</param>
	/// <param name="blockSize">Block size of all blocks in this database</param>
	Database(std::string databaseName, int blockSize = 1024)
	{
		name_ = databaseName;
		blockSize_ = blockSize;
	}

	~Database();

	static const std::unordered_map<std::string, std::shared_ptr<Database>>& LoadedDatabases() { return loadedDatabases_; }
	std::string& const GetName() { return name_; }
	int GetBlockSize() const { return blockSize_; }
	const std::unordered_map<std::string, Table>& GetTables() const { return tables_; }
	static void AddToInMemoryDatabaseList(std::shared_ptr<Database> database) { loadedDatabases_.insert({ database->name_, database }); }
	static std::shared_ptr<Database>& GetDatabaseByName(std::string databaseName) { return loadedDatabases_[databaseName]; }
	static void DestroyDatabase(std::string databaseName) { loadedDatabases_.erase(databaseName); }
	static std::shared_ptr<Database>& LoadDatabase(std::string fileDbName, std::string path);
	static void LoadColumns(std::string path, std::string dbName, Table table, std::vector<std::string> columnNames);
	static void LoadDatabasesFromDisk();
	static void SaveAllToDisk(std::string path);
	Table& CreateTable(std::unordered_map<std::string, ColumnType> columns, std::string tableName);
	void Persist(std::string path);
};

