#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include "DataType.h"
#include "QueryEngine/Context.h"
#include "Table.h"
/// <summary>
/// The main class representing database containing tables with data.
/// </summary>

class IColumn;

class Database
{
    friend class DatabaseGenerator;

private:
    static std::mutex dbAccessMutex_;
    std::string name_;
    int32_t blockSize_;
    std::unordered_map<std::string, Table> tables_;

    /// <summary>
    /// Load column of a table into memory from disk.
    /// </summary>
    /// <param name="path">Path directory, where column file (*.col) is.</param>
    /// <param name="dbName">Name of the database.</param>
    /// <param name="persistenceFormatVersion">Version of format used to persist .db and .col files
    /// into disk.</param> <param name="table">Instance of table into which the column should be
    /// added.</param> <param name="columnName">Names of particular column.</param>
    static void LoadColumn(const char* path,
                           const char* dbName,
                           int32_t persistenceFormatVersion,
                           Table& table,
                           const std::string& columnName);

    /// <summary>
    /// Write column into disk.
    /// </summary>
    /// <param name="column">Column to be written.</param>
    /// <param name="pathStr">Path to database storage directory.</param>
    /// <param name="name">Names of particular column.</param>
    /// <param name="table">Names of particular table.</param>
    static void WriteColumn(const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                            std::string pathStr,
                            std::string name,
                            const std::pair<const std::string, Table>& table);

public:
    static constexpr const char* SEPARATOR = "@";
    static constexpr const int32_t PERSISTENCE_FORMAT_VERSION = 1;
    static std::mutex dbMutex_;
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
    /// </summary>
    /// <param name="databaseName">Database name.</param>
    /// <param name="blockSize">Block size of all blocks in this database.</param>
    Database(const char* databaseName, int32_t blockSize = 1 << 18);

    ~Database();

    // getters:
    const std::string& GetName() const
    {
        return name_;
    }
    int GetBlockSize() const
    {
        return blockSize_;
    }
    std::unordered_map<std::string, Table>& GetTables()
    {
        return tables_;
    }
    static bool Exists(const std::string& databaseName)
    {
        return Context::getInstance().GetLoadedDatabases().find(databaseName) !=
               Context::getInstance().GetLoadedDatabases().end();
    }
    static std::vector<std::string> GetDatabaseNames();

    /// <summary>
    /// Set saveNecessaty_ to false for block, column and table, because data in the database were NOT modified yet.
    /// </summary>
    void SetSaveNecessaryToFalseForEverything();

    /// <summary>
    /// Save only .db file to disk.
    /// </summary>
    /// <param name="path">Path to database storage directory.</param>
    void PersistOnlyDbFile(const char* path);

    /// <summary>
    /// Save database from memory to disk.
    /// </summary>
    /// <param name="path">Path to database storage directory.</param>
    void Persist(const char* path);

    /// <summary>
    /// Save modified blocks and columns of the database from memory to disk.
    /// </summary>
    /// <param name="path">Path to database storage directory.</param>
    void PersistOnlyModified(const char* path);

    /// <summary>
    /// Save all databases currently in memory to disk. All databases will be saved in the same directory.
    /// </summary>
    static void SaveAllToDisk();

    /// <summary>
    /// Save only modified blocks and columns to disk. All databases will be saved in the same directory.
    /// </summary>
    static void SaveModifiedToDisk();

    /// <summary>
    /// Load databases from disk storage. Databases .db and .col files have to be in the same directory,
    /// so all databases have to be in the same directory to be loaded using this procedure.
    /// </summary>
    static void LoadDatabasesFromDisk();

    /// <summary>
    /// Delete database from disk. Deletes .db and .col files which belong to the specified
    /// database. Database is not deleted from memory.
    /// </summary>
    void DeleteDatabaseFromDisk();

    /// <summary>
    /// <param name="tableName">Name of the table to be deleted.</param>
    /// Delete table from disk. Deletes .col files which belong to the specified table of currently loaded database.
    /// To alter .db file, this action also calls a function PersistOnlyDbFile().
    /// Table needs to be deleted from memory before calling this method, so that .db file can be updated correctly.
    /// </summary>
    void DeleteTableFromDisk(const char* tableName);

    /// <summary>
    /// <param name="tableName">Name of the table which have the specified column that will be
    /// deleted.</param> <param name="columnName">Name of the column file (*.col) without the ".col"
    /// suffix that will be deleted.</param> Delete column of a table. Deletes single .col file
    /// which belongs to specified column and specified table. To alter .db file, this action also
    /// calls a function Persist. Column needs to be deleted from memory before calling this method,
    /// so that .db file can be updated correctly.
    /// </summary>
    void DeleteColumnFromDisk(const char* tableName, const char* columnName);

    /// <summary>
    /// Load database from disk into memory.
    /// </summary>
    /// <param name="fileDbName">Name of the database file (*.db) without the ".db" suffix.</param>
    /// <param name="path">Path to directory in which database files are.</param>
    /// <returns>Newly created table.</returns>
    static std::shared_ptr<Database> LoadDatabase(const char* fileDbName, const char* path);

    /// <summary>
    /// Creates table with given name and columns and adds it to database. If the table already
    /// existed, create missing columns if there are any missing
    /// </summary>
    /// <param name="columns">Columns with types.</param>
    /// <param name="tableName">Table name.</param>
    /// <param name="areNullable">Nullablity of columns. Default values are set to be true.</param>
    /// <returns>Newly created table.</returns>
    Table&
    CreateTable(const std::unordered_map<std::string, DataType>& columns,
                const char* tableName,
                const std::unordered_map<std::string, bool>& areNullable = std::unordered_map<std::string, bool>());

    /// <summary>
    /// Add database to in memory list.
    /// </summary>
    /// <param name="database">Database to be added.</param>
    static void AddToInMemoryDatabaseList(std::shared_ptr<Database> database);

    /// <summary>
    /// Get database from in memory list.
    /// </summary>
    /// <param name="databaseName">Name of database to get.</param>
    /// <returns>Database object or null-</returns>
    static std::shared_ptr<Database> GetDatabaseByName(std::string databaseName)
    {
        std::lock_guard<std::mutex> lock(dbAccessMutex_);
        try
        {
            return Context::getInstance().GetLoadedDatabases().at(databaseName);
        }
        catch (std::out_of_range&)
        {
            return nullptr;
        }
    }

    /// <summary>
    /// Remove database from in memory database list.
    /// </summary>
    /// <param name="databaseName">Name of database to be removed.</param>
    static void RemoveFromInMemoryDatabaseList(const char* databaseName);
};
