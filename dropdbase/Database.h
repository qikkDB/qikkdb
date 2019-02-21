#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include "DataType.h"
#include "Table.h"


/// <summary>
/// The main class representing database containing tables with data
/// </summary>
class Database
{
	friend class DatabaseGenerator;

private:
	static std::unordered_map<std::string, std::shared_ptr<Database>> loadedDatabases_;
	static std::mutex dbMutex_;
	std::string name_;
	int32_t blockSize_;
	std::unordered_map<std::string, Table> tables_;

public:
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
	/// </summary>
	/// <param name="databaseName">Database name.</param>
	/// <param name="blockSize">Block size of all blocks in this database</param>
	Database(const char* databaseName, int32_t blockSize = 1 << 18);

	~Database();

	//getters:
	const std::string& GetName() const { return name_; }
	int GetBlockSize() const { return blockSize_; }
	std::unordered_map<std::string, Table>& GetTables() { return tables_; }
	static bool Exists(const std::string& databaseName) { return loadedDatabases_.find(databaseName) != loadedDatabases_.end(); }
	static std::vector<std::string> GetDatabaseNames();
	/// <summary>
	/// Save database from memory to disk.
	/// </summary>
	/// <param name="path">Path to database storage directory</param>
	void Persist(const char* path);

	/// <summary>
	/// Save all databases currently in memory to disk. All databases will be saved in the same directory
	/// </summary>
	static void SaveAllToDisk();

	/// <summary>
	/// Load databases from disk storage. Databases .db and .col files have to be in the same directory,
	/// so all databases have to be in the same directory to be loaded using this procedure
	/// </summary>
	static void LoadDatabasesFromDisk();

	/// <summary>
	/// Load database from disc into memory.
	/// </summary>
	/// <param name="fileDbName">Name of the database file (*.db) without the ".db" suffix.</param>
	/// <param name="path">Path to directory in which database files are.</param>
	static std::shared_ptr<Database> LoadDatabase(const char* fileDbName, const char* path);

	/// <summary>
	/// Load columns of a table into memory from disc.
	/// </summary>
	/// <param name="path">Path directory, where column files (*.col) are.</param>
	/// <param name="table">Instance of table into which the columns should be added.</param>
	/// <param name="columnNames">Names of particular columns.</param>
	static void LoadColumns(const char* path, const char* dbName, Table& table, const std::vector<std::string>& columnNames);

	/// <summary>
	/// Creates table with given name and columns and adds it to database. If the table already existed, create missing columns if there are any missing
	/// </summary>
	/// <param name="columns">Columns with types.</param>
	/// <param name="tableName">Table name.</param>
	/// <returns>Newly created table</returns>
	Table& CreateTable(const std::unordered_map<std::string, DataType>& columns, const char* tableName);

	/// <summary>
	/// Add database to in memory list
	/// </summary>
	/// <param name="database">Database to add</param>
	static void AddToInMemoryDatabaseList(std::shared_ptr<Database> database);

	/// <summary>
	/// Get database from in memory list
	/// </summary>
	/// <param name="databaseName">Name of database to get</param>
	/// <returns>Database object or null</returns>
	static std::shared_ptr<Database> GetDatabaseByName(std::string databaseName)
	{
		std::lock_guard<std::mutex> lock(dbMutex_);
		try
		{
			return loadedDatabases_.at(databaseName);
		}
		catch(std::out_of_range&)
		{
			return nullptr;
		}
	}

	/// <summary>
	/// Remove database from in memory list
	/// </summary>
	/// <param name="databaseName">Name of database to be removed</param>
	static void DestroyDatabase(const char* databaseName) { std::lock_guard<std::mutex> lock(dbMutex_); loadedDatabases_.erase(databaseName); }

	/// <summary>
	/// Get number of blocks
	/// </summary>
	/// <returns>Number of blocks</param>
	int GetBlockCount();
};
