#pragma once
#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include "DataType.h"
#include "ConstraintType.h"
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
    /// <param name="fileDbPath">Path to DB_EXTENSION file.</param>
    /// <param name="fileAddressPath">Path to COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="fileDataPath">Path to COLUMN_DATA_EXTENSION file.</param>
    /// <param name="fileFragmentPath">Path to FRAGMENT_DATA_EXTENSION file.</param>
    /// <param name="encoding">Encoding of the string data in FRAGMENT_DATA_EXTENSION file.</param>
    /// <param name="persistenceFormatVersion">Version of format used to persist DB_EXTENSION and
    /// COLUMN_DATA_EXTENSION files into disk.</param> <param name="type">Type of column according
    /// to DataType enumeration.</param>
    /// <param name="isNullable">Flag if a column can have NULL values.</param>
    /// <param name="isUnique">Flag if a column can have only unique values and not
    /// a single one NULL value.</param>
    /// <param name="defaultValue">Default column value in string format.</param>
    ///< param name="table">Instance of table into which the column
    /// should be added.</param> <param name="columnName">Names of particular column.</param>
    static void LoadColumn(const std::string fileDbPath,
                           const std::string fileAddressPath,
                           const std::string fileDataPath,
                           const std::string fileFragmentPath,
                           const std::string encoding,
                           const int32_t persistenceFormatVersion,
                           const int32_t type,
                           const bool isNullable,
                           const bool isUnique,
                           const std::string defaultValue,
                           Table& table,
                           const std::string columnName);

    /// <summary>
    /// Write single block into disk. It has to seek the block's position in the
    /// COLUMN_DATA_EXTENSION file and replace the block's data with the data wich is in memory.
    /// </summary>
    /// <param name="table">Name of the particular table.</param>
    /// <param name="column">Name of the column to which the block belongs to.</param>
    static void WriteBlock(const Table& table,
                           const std::pair<const std::string, std::unique_ptr<IColumn>>& column);

    /// <summary>
    /// Write column into disk (all it's blocks).
    /// </summary>
    /// <param name="column">Column to be written.</param>
    /// <param name="dbName">Name of the database.</param>
    /// <param name="table">Name of the particular table.</param>
    static void WriteColumn(const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                            const std::string dbName,
                            const Table& table);

public:
    static constexpr const int32_t PERSISTENCE_FORMAT_VERSION = 1;
    static constexpr const char* SEPARATOR = "@";
    static constexpr const char* DB_EXTENSION = ".db";
    static constexpr const char* COLUMN_DATA_EXTENSION = ".data";
    static constexpr const char* COLUMN_ADDRESS_EXTENSION = ".adrs";
    static constexpr const char* FRAGMENT_DATA_EXTENSION = ".fragdata";
    static constexpr const int32_t FRAGMENT_SIZE_BYTES = 1048576;
    static constexpr const char* POLYGON_DEFAULT_VALUE = "POLYGON((0 0, 1 1, 2 2, 0 0))";
    static std::mutex dbMutex_;
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
    /// </summary>
    /// <param name="databaseName">Database name.</param>
    /// <param name="blockSize">Block size of all blocks in this database.</param>
    Database(const char* databaseName, int32_t blockSize = 1 << 18);

    ~Database();

    void SetName(const std::string& newDatabaseName);

    const std::string& GetName() const
    {
        return name_;
    }

    /// <summary>
    /// Returns the deault database's block size. This blockSize is used (it's value), when creating
    /// a new table which does not have a specified blockSize.
    /// </summary>
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

	/// <summary>
    /// Returnes names of the loaded databases in memory.
    /// </summary>
    static std::vector<std::string> GetDatabaseNames();

    /// <summary>
    /// Copy block of data from the specified column of the source table and insert it in the column with the same name in the destination table.
    /// This is used when block sizes of those two tables are different.
    /// </summary>
    /// <param name="srcTable">Source table.</param>
    /// <param name="destTable">Destination table.</param>
    /// <param name="columnName">Name of the column which data will be copied.</param>
    static void CopyBlocksOfColumn(Table& srcTable, Table& destTable, const std::string& columnName);

    /// <summary>
    /// Save modified blocks of data of all loaded database to disk.
    /// </summary>
    static void SaveModifiedToDisk();

    /// <summary>
    /// Save only DB_EXTENSION file to disk into directory defined in configuration file.
    /// </summary>
    void PersistOnlyDbFile();

    /// <summary>
    /// Save database from memory to disk. Rewrites all database files.
    /// </summary>
    void Persist();

    /// <summary>
    /// Save modified blocks of the specified table from memory to disk.
    /// </summary>
    /// <param name="tableName">Name of the table which modified blocks of data will be saved.</param>
    void PersistOnlyModified(const std::string tableName);

    /// <summary>
    /// Save all databases currently loaded in memory to disk. Rewrites all loaded databases' files.
    /// </summary>
    static void SaveAllToDisk();

    /// <summary>
    /// Load databases from disk storage.
    /// </summary>
    static void LoadDatabasesFromDisk();

    /// <summary>
    /// Rename specified table.
    /// </summary>
    /// <param name="oldTableName">Table name of the table which will be renamed.</param>
    /// <param name="oldTableName">New table name to which the table will be renamed.</param>
    void RenameTable(const std::string& oldTablename, const std::string& newTableName);

    /// <summary>
    /// Delete database from disk. Deletes DB_EXTENSION and COLUMN_DATA_EXTENSION files which belong
    /// to the specified database. Database is not deleted from memory.
    /// </summary>
    void DeleteDatabaseFromDisk();

    /// <summary>
    /// <param name="tableName">Name of the table to be deleted.</param>
    /// Delete table from disk. Deletes COLUMN_DATA_EXTENSION files which belong to the specified
    /// table of currently loaded database. To alter DB_EXTENSION file, this action also calls a
    /// function PersistOnlyDbFile(). Table needs to be deleted from memory before calling this
    /// method, so that DB_EXTENSION file can be updated correctly.
    /// </summary>
    void DeleteTableFromDisk(const char* tableName);

    /// <summary>
    /// <param name="tableName">Name of the table which have the specified column that will be
    /// deleted.</param> <param name="columnName">Name of the COLUMN_DATA_EXTENSION file without the COLUMN_DATA_EXTENSION
    /// suffix that will be deleted.</param> Delete column of a table. Deletes single COLUMN_DATA_EXTENSION file
    /// which belongs to specified column and specified table. To alter DB_EXTENSION file, this action also
    /// calls a function Persist. Column needs to be deleted from memory before calling this method,
    /// so that DB_EXTENSION file can be updated correctly.
    /// </summary>
    void DeleteColumnFromDisk(const char* tableName, const char* columnName);

    /// <summary>
    /// Changes the block size of all tables of database and all the columns of the all tables will be affected - their blocks
    /// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
    /// </summary>
    /// <param name="newBlockSize">New block size of all tables and columns of this database.</param>
    void ChangeDatabaseBlockSize(int32_t newBlockSize);

    /// <summary>
    /// Changes the block size of a table and all the columns of the table will be affected - their blocks
    /// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
    /// </summary>
    /// <param name="tableName">Name of the table which have which block size will be changed.</param>
    /// <param name="newBlockSize">New block size of a table and columns of this table.</param>
    void ChangeTableBlockSize(const std::string tableName, int32_t newBlockSize);

    /// <summary>
    /// Load database from disk into memory.
    /// </summary>
    /// <param name="fileDbName">Name of the database file without the DB_EXTENSION suffix.</param>
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
    /// <param name="blockSize">Table block size.</param>
    /// <returns>Newly created table.</returns>
    Table&
    CreateTable(const std::unordered_map<std::string, DataType>& columns,
                const char* tableName,
                const std::unordered_map<std::string, bool>& areNullable = std::unordered_map<std::string, bool>(),
                const std::unordered_map<std::string, bool>& areUnique = std::unordered_map<std::string, bool>(),
                int32_t blockSize = -1);

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
