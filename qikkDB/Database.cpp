#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <cstdint>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <stdio.h>
#include <algorithm>

#include "ColumnBase.h"
#include "Configuration.h"
#include "Database.h"
#include "Table.h"
#include "Types/ComplexPolygon.pb.h"
#include "QueryEngine/Context.h"

std::mutex Database::dbMutex_;
std::mutex Database::dbAccessMutex_;
std::mutex Database::dbFilesMutex_;

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
/// </summary>
/// <param name="databaseName">Database name.</param>
/// <param name="blockSize">Block size of all blocks in this database.</param>
Database::Database(const char* databaseName, int32_t blockSize)
{
    name_ = databaseName;
    blockSize_ = blockSize;
}

Database::~Database()
{
    Context& context = Context::getInstance();
    int32_t oldDeviceID = context.getBoundDeviceID();
    // Clear cache for all devices
    for (int32_t deviceID = 0; deviceID < Context::getInstance().getDeviceCount(); deviceID++)
    {
        context.bindDeviceToContext(deviceID);
        GPUMemoryCache& cacheForDevice = Context::getInstance().getCacheForDevice(deviceID);
        for (auto const& table : tables_)
        {
            for (auto const& column : table.second.GetColumns())
            {
                int32_t blockCount = column.second.get()->GetBlockCount();
                for (int32_t i = 0; i < blockCount; i++)
                {
                    cacheForDevice.clearCachedBlock(
                        name_, table.second.GetName() + "." + column.second.get()->GetName(), i);
                    if (column.second.get()->GetIsNullable())
                    {
                        cacheForDevice.clearCachedBlock(name_,
                                                        table.second.GetName() + "." +
                                                            column.second.get()->GetName() + "_nullMask",
                                                        i);
                    }
                }
            }
        }
    }
    context.bindDeviceToContext(oldDeviceID);
}

void Database::SetName(const std::string& newDatabaseName)
{
    name_ = newDatabaseName;
}

std::vector<std::string> Database::GetDatabaseNames()
{
    std::vector<std::string> ret;
    for (auto& entry : Context::getInstance().GetLoadedDatabases())
    {
        ret.push_back(entry.first);
    }
    return ret;
}

/// <summary>
/// Copy block of data from the specified column of the source table and insert it in the column with the same name in the destination table.
/// This is used when block sizes of those two tables are different.
/// </summary>
/// <param name="srcTable">Source table.</param>
/// <param name="destTable">Destination table.</param>
/// <param name="columnName">Name of the column whose data will be copied.</param>
void Database::CopyBlocksOfColumn(Table& srcTable, Table& dstTable, const std::string& columnName)
{
    BOOST_LOG_TRIVIAL(debug) << "Copying data (column name: " << columnName
                             << ") from the table named: " << srcTable.GetName()
                             << " to the table named: " << dstTable.GetName() << " has started.";

    auto columnType = srcTable.GetColumns().find(columnName)->second.get()->GetColumnType();

    switch (columnType)
    {
    case COLUMN_POLYGON:
    {
        auto& column = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(
            *srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(
            *dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<ColmnarDB::Types::ComplexPolygon>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_POINT:
    {
        auto& column =
            dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn =
            dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<ColmnarDB::Types::Point>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_STRING:
    {
        auto& column = dynamic_cast<ColumnBase<std::string>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<std::string>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<std::string>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_INT8_T:
    {
        auto& column = dynamic_cast<ColumnBase<int8_t>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<int8_t>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<int8_t>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_INT:
    {
        auto& column = dynamic_cast<ColumnBase<int32_t>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<int32_t>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<int32_t>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_LONG:
    {
        auto& column = dynamic_cast<ColumnBase<int64_t>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<int64_t>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<int64_t>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_FLOAT:
    {
        auto& column = dynamic_cast<ColumnBase<float>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<float>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<float>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    case COLUMN_DOUBLE:
    {
        auto& column = dynamic_cast<ColumnBase<double>&>(*srcTable.GetColumns().at(columnName));
        auto& dstColumn = dynamic_cast<ColumnBase<double>&>(*dstTable.GetColumns().at(columnName));
        const std::vector<BlockBase<double>*> blocks = column.GetBlocksList();
        dstColumn.ResizeColumn(&column);
        break;
    }
    default:
        throw std::runtime_error("Unsupported column type.");
        break;
    }

    BOOST_LOG_TRIVIAL(debug) << "Copying data (column name: " << columnName
                             << ") from the table named: " << srcTable.GetName()
                             << " to the table named: " << dstTable.GetName() << " has finished.";
}

/// <summary>
/// Set saveNecessaty_ to false for block, column and table, because data in the database were NOT modified yet.
/// </summary>
void Database::SetSaveNecessaryToFalseForEverything()
{
    auto& tables = GetTables();

    for (auto& table : tables)
    {
        table.second.SetSaveNecessaryToFalse();

        auto& columns = table.second.GetColumns();

        for (const auto& column : columns)
        {
            auto type = column.second.get()->GetColumnType();

            switch (type)
            {
            case COLUMN_POLYGON:
            {
                auto castedColumn =
                    dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_POINT:
            {
                auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_STRING:
            {
                auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_INT8_T:
            {
                auto castedColumn = dynamic_cast<ColumnBase<int8_t>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_INT:
            {
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_LONG:
            {
                auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_FLOAT:
            {
                auto castedColumn = dynamic_cast<ColumnBase<float>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            case COLUMN_DOUBLE:
            {
                auto castedColumn = dynamic_cast<ColumnBase<double>*>(column.second.get());
                castedColumn->SetSaveNecessaryToFalse();

                auto& blocks = castedColumn->GetBlocksList();

                for (int32_t i = 0; i < blocks.size(); i++)
                {
                    blocks[i]->SetSaveNecessaryToFalse();
                }
            }
            break;

            default:
                throw std::runtime_error("Unsupported column type.");
                break;
            }
        }
    }
}

/// <summary>
/// Save only .db file to disk.
/// </summary>
/// <param name="path">Path to database storage directory.</param>
void Database::PersistOnlyDbFile(const char* path)
{
    auto& tables = GetTables();
    auto& name = GetName();
    auto pathStr = std::string(path);

    boost::filesystem::create_directories(path);

    int32_t blockSize = GetBlockSize();
    int32_t tableSize = tables.size();

    // write file .db
    BOOST_LOG_TRIVIAL(debug) << "Saving .db file with name: " << pathStr << name << ".db";
    std::ofstream dbFile(pathStr + "/" + name + ".db", std::ios::binary);

    if (dbFile.is_open())
    {
        int32_t dbNameLength = name.length() + 1; // +1 because '\0'

        dbFile.write(reinterpret_cast<const char*>(&PERSISTENCE_FORMAT_VERSION),
                     sizeof(int32_t)); // write persistence format version
        dbFile.write(reinterpret_cast<char*>(&dbNameLength), sizeof(int32_t)); // write db name length
        dbFile.write(name.c_str(), dbNameLength); // write db name
        dbFile.write(reinterpret_cast<char*>(&blockSize), sizeof(int32_t)); // write block size
        dbFile.write(reinterpret_cast<char*>(&tableSize), sizeof(int32_t)); // write number of tables
        for (auto& table : tables)
        {
            auto& columns = table.second.GetColumns();
            const auto& sortingColumns = table.second.GetSortingColumns();
            int32_t tableNameLength = table.first.length() + 1; // +1 because '\0'
            int32_t columnNumber = columns.size();
            int32_t sortingColumnNumber = sortingColumns.size();
            int32_t tableBlockSize = table.second.GetBlockSize();

            dbFile.write(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); // write table name length
            dbFile.write(table.first.c_str(), tableNameLength); // write table name
            dbFile.write(reinterpret_cast<char*>(&tableBlockSize),
                         sizeof(int32_t)); // write number of columns of the table
            dbFile.write(reinterpret_cast<char*>(&columnNumber), sizeof(int32_t)); // write number of columns of the table
            dbFile.write(reinterpret_cast<char*>(&sortingColumnNumber),
                         sizeof(int32_t)); // write number of sorting columns of the table

            if (sortingColumnNumber > 0)
            {
                for (const std::string sortingColumn : sortingColumns)
                {
                    int32_t sortingColumnLength = sortingColumn.length() + 1; // +1 because '\0'

                    dbFile.write(reinterpret_cast<char*>(&sortingColumnLength),
                                 sizeof(int32_t)); // write sorting column name length
                    dbFile.write(sortingColumn.c_str(), sortingColumnLength); // write sorting column name
                    BOOST_LOG_TRIVIAL(debug)
                        << "Sorting column (table: " + std::string(table.first.c_str()) +
                               ") saved: " + std::string(sortingColumn.c_str()) + ".";
                }
            }

            for (const auto& column : columns)
            {
                int32_t columnNameLength = column.first.length() + 1; // +1 because '\0'

                dbFile.write(reinterpret_cast<char*>(&columnNameLength), sizeof(int32_t)); // write column name length
                dbFile.write(column.first.c_str(), columnNameLength); // write column name
            }
        }
        dbFile.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(error)
            << "Could not open file " + std::string(pathStr + "/" + name + ".db") +
                   " for writing. Persisting .db file was not successful. Check if the process "
                   "have write access into the folder or file.";
    }
}

/// <summary>
/// Save database from memory to disk.
/// </summary>
/// <param name="path">Path to database storage directory.</param>
void Database::Persist(const char* path)
{
    auto& tables = GetTables();
    auto& name = GetName();
    auto pathStr = std::string(path);

    BOOST_LOG_TRIVIAL(info) << "Saving database with name: " << name << " and " << tables.size() << " tables.";

    int32_t blockSize = GetBlockSize();
    int32_t tableSize = tables.size();

    PersistOnlyDbFile(path);

    // write files .col:
    for (auto& table : tables)
    {
        auto& columns = table.second.GetColumns();

        std::vector<std::thread> threads;

        for (const auto& column : columns)
        {
            threads.emplace_back(Database::WriteColumn, std::ref(column), pathStr, name, std::ref(table));
            column.second.get()->SetSaveNecessaryToFalse();
        }

        for (int j = 0; j < columns.size(); j++)
        {
            threads[j].join();
        }

        table.second.SetSaveNecessaryToFalse();
    }

    if (boost::filesystem::exists(boost::filesystem::path(path + name + ".db")))
    {
        BOOST_LOG_TRIVIAL(info) << "Database " << name << " was successfully saved into disk.";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info)
            << "Database "
            << name << " was NOT saved into disk. Check if you have write access into the destination folder.";
    }
}

/// <summary>
/// Save modified blocks and columns of the database from memory to disk.
/// </summary>
/// <param name="path">Path to database storage directory.</param>
void Database::PersistOnlyModified(const char* path)
{
    auto& tables = GetTables();
    auto& name = GetName();
    auto pathStr = std::string(path);

    BOOST_LOG_TRIVIAL(info) << "Saving database with name: " << name << " and " << tables.size() << " table/s.";

    int32_t blockSize = GetBlockSize();
    int32_t tableSize = tables.size();

    // always persist at least db file
    PersistOnlyDbFile(path);

    // write files .col:
    for (auto& table : tables)
    {
        if (table.second.GetSaveNecessary())
        {
            const auto& columns = table.second.GetColumns();

            std::vector<std::thread> threads;

            for (auto& column : columns)
            {
                if (column.second.get()->GetSaveNecessary())
                {
                    threads.emplace_back(Database::WriteColumn, std::ref(column), pathStr, name,
                                         std::ref(table));
                    column.second.get()->SetSaveNecessaryToFalse();
                }
            }

            for (int j = 0; j < threads.size(); j++)
            {
                threads[j].join();
            }

            table.second.SetSaveNecessaryToFalse();
        }
    }

    if (boost::filesystem::exists(boost::filesystem::path(path + name + ".db")))
    {
        BOOST_LOG_TRIVIAL(info) << "Database " << name << " was successfully saved into disk.";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info)
            << "Database "
            << name << " was NOT saved into disk. Check if you have write access into the destination folder.";
    }
}

/// <summary>
/// Save all databases currently in memory to disk. All databases will be saved in the same directory
/// </summary>
void Database::SaveAllToDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    BOOST_LOG_TRIVIAL(info) << "Saving all loaded databases to disk has started...";
    auto path = Configuration::GetInstance().GetDatabaseDir().c_str();
    for (auto& database : Context::getInstance().GetLoadedDatabases())
    {
        database.second->Persist(path);
    }
    BOOST_LOG_TRIVIAL(info) << "Saving loaded databases to disk has finished.";
}

/// <summary>
/// Save only modified blocks and columns to disk. All databases will be saved in the same directory.
/// </summary>
void Database::SaveModifiedToDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    BOOST_LOG_TRIVIAL(info)
        << "Saving only modified blocks and columns of the loaded databases to disk has started...";
    auto path = Configuration::GetInstance().GetDatabaseDir().c_str();
    for (auto& database : Context::getInstance().GetLoadedDatabases())
    {
        database.second->PersistOnlyModified(path);
    }
    BOOST_LOG_TRIVIAL(info)
        << "Saving only modified blocks and columns of the loaded databases to disk has finished.";
}

/// <summary>
/// Load databases from disk storage. Databases .db and .col files have to be in the same directory,
/// so all databases have to be in the same dorectory to be loaded using this procedure.
/// </summary>
void Database::LoadDatabasesFromDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    if (boost::filesystem::exists(path))
    {
        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            auto extension = p.path().extension();
            if (extension == ".db")
            {
                auto database =
                    Database::LoadDatabase(p.path().filename().stem().generic_string().c_str(), path.c_str());

                Context::getInstance().CheckDatabasesLimit(
                    Context::getInstance().GetLoadedDatabases().size());

                if (database != nullptr)
                {
                    database->SetSaveNecessaryToFalseForEverything();
                    Context::getInstance().GetLoadedDatabases().insert({database->name_, database});
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Directory " << path << " does not exists.";
    }
}

/// <summary>
/// Rename specified table.
/// </summary>
/// <param name="oldTableName">Table name of the table which will be renamed.</param>
/// <param name="oldTableName">New table name to which the table will be renamed.</param>
void Database::RenameTable(const std::string& oldTableName, const std::string& newTableName)
{
    tables_.at(oldTableName).SetTableName(newTableName);
    auto handler = tables_.extract(oldTableName);
    handler.key() = newTableName;
    tables_.insert(std::move(handler));

    auto& path = Configuration::GetInstance().GetDatabaseDir();

    if (boost::filesystem::remove(path + name_ + ".db"))
    {
        std::string prefix(name_ + SEPARATOR + oldTableName + SEPARATOR);
        std::string prefix2(path + name_ + SEPARATOR + oldTableName + SEPARATOR);

        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            // rename files which starts with prefix of db name and table name:
            if (!p.path().string().compare(path.size(), prefix.size(), prefix))
            {

                std::string columnName = p.path().string().substr(prefix2.size());
                const boost::filesystem::path& newPath{path + name_ + SEPARATOR + newTableName +
                                                       SEPARATOR + columnName};
                boost::filesystem::rename(p.path(), newPath);
            }
        }

        PersistOnlyDbFile(path.c_str());
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning)
            << "Renaming table: Main (.db) file of db " << name_
            << " was NOT removed from disk. No such file (if the database was not yet saved, "
               "ignore this warning) or no write access.";
    }
}

/// <summary>
/// Delete database from disk. Deletes .db and .col files which belong to the specified database.
/// Database is not deleted from memory.
/// </summary>
void Database::DeleteDatabaseFromDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    // std::cout << "DeleteDatabaseFromDisk path: " << path << std::endl;
    if (boost::filesystem::exists(path))
    {
        // Delete main .db file
        if (boost::filesystem::remove(path + name_ + ".db"))
        {
            BOOST_LOG_TRIVIAL(info) << "Main (.db) file of db " << name_ << " was successfully removed from disk.";
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Main (.db) file of db "
                                       << name_ << " was NOT removed from disk. No such file or write access.";
        }

        // Delete tables and columns
        std::string prefix(path + name_ + SEPARATOR);
        // Replace backslash with slash for comparison reasons
        std::replace(prefix.begin(), prefix.end(), '\\', '/');

        // Iterate through all files in database directory
        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            // Replace backslash with slash for comparison with prefix
            std::string columnPath = p.path().string();
            std::replace(columnPath.begin(), columnPath.end(), '\\', '/');

            // Delete files which starts with prefix of db name:
            if (!columnPath.compare(0, prefix.size(), prefix))
            {
                if (boost::filesystem::remove(p.path().string().c_str()))
                {
                    BOOST_LOG_TRIVIAL(info) << "File " << columnPath << " was successfully removed from disk.";
                }
                else
                {
                    BOOST_LOG_TRIVIAL(warning)
                        << "File " << columnPath
                        << " was NOT removed from disk. No such file or write access.";
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Directory " << path << " does not exists.";
    }
}

/// <summary>
/// <param name="tableName">Name of the table to be deleted.</param>
/// Delete table from disk. Deletes .col files which belong to the specified table of currently loaded database.
/// To alter .db file, this action also calls a function PersistOnlyDbFile().
/// Table needs to be deleted from memory before calling this method, so that .db file can be updated correctly.
/// </summary>
void Database::DeleteTableFromDisk(const char* tableName)
{
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    if (boost::filesystem::exists(path))
    {
        std::string prefix(path + name_ + SEPARATOR + std::string(tableName) + SEPARATOR);
        // Replace backslash with slash for comparison reasons
        std::replace(prefix.begin(), prefix.end(), '\\', '/');

        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            // Replace backslash with slash for comparison with prefix
            std::string columnPath = p.path().string();
            std::replace(columnPath.begin(), columnPath.end(), '\\', '/');

            // delete files which starts with prefix of db name and table name:
            if (!columnPath.compare(0, prefix.size(), prefix))
            {
                if (boost::filesystem::remove(p.path().string().c_str()))
                {
                    BOOST_LOG_TRIVIAL(info) << "File " << columnPath << " from database " << name_
                                            << " was successfully removed from disk.";
                }
                else
                {
                    BOOST_LOG_TRIVIAL(warning)
                        << "File " << columnPath
                        << " was NOT removed from disk. No such file or write access.";
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Directory " << path << " does not exists.";
    }

    // persist only db file, so that changes are saved, BUT PERSIST ONLY if there already is a .db file, so it is not only in memory
    if (boost::filesystem::exists(path + name_ + ".db"))
    {
        PersistOnlyDbFile(Configuration::GetInstance().GetDatabaseDir().c_str());
    }
}

/// <summary>
/// <param name="tableName">Name of the table which have the specified column that will be deleted.</param>
/// <param name="columnName">Name of the column file (*.col) without the ".col" suffix that will be deleted.</param>
/// Delete column of a table. Deletes single .col file which belongs to specified column and specified table.
/// To alter .db file, this action also calls a function Persist.
/// Column needs to be deleted from memory before calling this method, so that .db file can be updated correctly.
/// </summary>
void Database::DeleteColumnFromDisk(const char* tableName, const char* columnName)
{
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    std::string filePath =
        path + name_ + SEPARATOR + std::string(tableName) + SEPARATOR + std::string(columnName) + ".col";

    if (boost::filesystem::exists(filePath))
    {
        if (boost::filesystem::remove(filePath.c_str()))
        {
            BOOST_LOG_TRIVIAL(info) << "Column " << columnName << " from table " << tableName << " from database "
                                    << name_ << " was successfully removed from disk.";
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "File " << filePath << " was NOT removed from disk. No such file or write access.";
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "File " << path << " does not exists.";
    }

    // persist only db file, so that changes are saved, BUT PERSIST ONLY if there already is a .db file, so it is not only in memory
    if (boost::filesystem::exists(path + name_ + ".db"))
    {
        PersistOnlyDbFile(Configuration::GetInstance().GetDatabaseDir().c_str());
    }
}

/// <summary>
/// Changes the block size of all tables of database and all the columns of the all tables will be affected - their blocks
/// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
/// </summary>
/// <param name="newBlockSize">New block size of all tables and columns of this database.</param>
void Database::ChangeDatabaseBlockSize(const int32_t newBlockSize)
{
    if (newBlockSize != blockSize_)
    {
        blockSize_ = newBlockSize;

        std::vector<std::thread> threads;

        for (auto& table : tables_)
        {
            threads.emplace_back(
                [&](Database* db) { db->ChangeTableBlockSize(table.first.c_str(), newBlockSize); }, this);
        }

        for (int32_t i = 0; i < threads.size(); i++)
        {
            threads[i].join();
        }
    }
}

/// <summary>
/// Changes the block size of a table and all the columns of the table will be affected - their blocks
/// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
/// </summary>
/// <param name="tableName">Name of the table which have which block size will be changed.</param>
/// <param name="newBlockSize">New block size of a table and columns of this table.</param>
void Database::ChangeTableBlockSize(const std::string tableName, const int32_t newBlockSize)
{
    if (newBlockSize != tables_.at(tableName).GetBlockSize())
    {
        auto& table = tables_.at(tableName);
        BOOST_LOG_TRIVIAL(info) << "The block size of the table named: " << tableName << " WILL BE changed from "
                                << table.GetBlockSize() << " to " << newBlockSize << ".";

        // create temporary table in memory with new block size
        tables_.emplace(std::make_pair(std::string("temp_" + tableName),
                                       Table(GetDatabaseByName(name_), ("temp_" + tableName).c_str(), newBlockSize)));

        auto newTableHashMap = tables_.find("temp_" + tableName);
        auto oldTableHashMap = tables_.find(tableName);

        // create the same columns in the new table (same as in the old table)
        if (newTableHashMap != tables_.end() && oldTableHashMap != tables_.end())
        {
            auto& newTable = newTableHashMap->second;
            auto& oldTable = oldTableHashMap->second;

            for (auto& column : oldTable.GetColumns())
            {
                newTable.CreateColumn(column.second->GetName().c_str(), column.second->GetColumnType(),
                                      column.second->GetIsNullable(), column.second->GetIsUnique());
            }
        }

        auto& columns = table.GetColumns();
        std::vector<std::thread> threads;
        for (auto& column : columns)
        {
            threads.emplace_back(Database::CopyBlocksOfColumn, std::ref(oldTableHashMap->second),
                                 std::ref(newTableHashMap->second), column.second.get()->GetName());
        }
        for (int j = 0; j < threads.size(); j++)
        {
            threads[j].join();
        }

        // delete (original) .col files with old block size which are persisted on disk, if they are persisted
        DeleteTableFromDisk(tableName.c_str());

        // initialize sorting Columns to be empty
        newTableHashMap->second.SetSortingColumns(std::vector<std::string>());

        // delete original table from memory
        tables_.erase(tableName);

        RenameTable(("temp_" + tableName).c_str(), tableName);

        // save all changes to disk
        Persist(Configuration::GetInstance().GetDatabaseDir().c_str());

        BOOST_LOG_TRIVIAL(info) << "The block size of the table named: " << tableName << " HAS BEEN changed from "
                                << table.GetBlockSize() << " to " << newBlockSize << ".";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info)
            << "The new block size of the table named: " << tableName
            << " was the same as the current block size, so it has not been changed.";
    }
}

/// <summary>
/// Load database from disk into memory.
/// </summary>
/// <param name="fileDbName">Name of the database file (*.db) without the ".db" suffix.</param>
/// <param name="path">Path to directory in which database files are.</param>
/// <returns>Shared pointer of database.</returns>
std::shared_ptr<Database> Database::LoadDatabase(const char* fileDbName, const char* path)
{
    const std::string filePath = std::string(path) + std::string(fileDbName) + ".db";

    // read file .db
    std::ifstream dbFile(filePath, std::ios::binary);

    dbFile.seekg(0, dbFile.end);
    const size_t fileSize = dbFile.tellg();
    if (fileSize != 0)
    {
        dbFile.seekg(0, dbFile.beg);
        BOOST_LOG_TRIVIAL(info) << "Loading database from: " << filePath << ".";

        int32_t persistenceFormatVersion;
        dbFile.read(reinterpret_cast<char*>(&persistenceFormatVersion),
                    sizeof(int32_t)); // read persistence format version

        if (persistenceFormatVersion != Database::PERSISTENCE_FORMAT_VERSION)
        {
            BOOST_LOG_TRIVIAL(warning)
                << "WARNING: Database persistence format version is different in database file: " << filePath
                << ". The persisted database files are in persistence format version: " << persistenceFormatVersion
                << " the current persistence format version in this version of database core is: "
                << Database::PERSISTENCE_FORMAT_VERSION
                << ". There is going to be coversion to the database core format verion. "
                << "The database files on disk will be changed after successful persistence.";
        }

        int32_t dbNameLength;
        dbFile.read(reinterpret_cast<char*>(&dbNameLength), sizeof(int32_t)); // read db name length

        std::unique_ptr<char[]> dbName(new char[dbNameLength]);
        dbFile.read(dbName.get(), dbNameLength); // read db name

        int32_t databaseBlockSize;
        dbFile.read(reinterpret_cast<char*>(&databaseBlockSize), sizeof(int32_t)); // read block size

        int32_t tablesCount;
        dbFile.read(reinterpret_cast<char*>(&tablesCount), sizeof(int32_t)); // read number of tables

        std::shared_ptr<Database> database = std::make_shared<Database>(dbName.get(), databaseBlockSize);

        for (int32_t i = 0; i < tablesCount; i++)
        {
            int32_t tableNameLength;
            dbFile.read(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); // read table name length

            std::unique_ptr<char[]> tableName(new char[tableNameLength]);
            dbFile.read(tableName.get(), tableNameLength); // read table name
            int32_t tableBlockSize;

            if (persistenceFormatVersion > 1)
            {
                dbFile.read(reinterpret_cast<char*>(&tableBlockSize), sizeof(int32_t)); // read table block size
            }
            else
            {
                tableBlockSize = databaseBlockSize;
            }

            BOOST_LOG_TRIVIAL(info)
                << "Block size for table: " + std::string(tableName.get()) +
                       " has been loaded and it's value is: " + std::to_string(tableBlockSize) + ".";

            Context::getInstance().CheckTablesLimit(database->tables_.size());

            database->tables_.emplace(std::make_pair(std::string(tableName.get()),
                                                     Table(database, tableName.get(), tableBlockSize)));

            int32_t columnCount;
            int32_t sortingColumnCount;
            dbFile.read(reinterpret_cast<char*>(&columnCount), sizeof(int32_t)); // read number of columns
            dbFile.read(reinterpret_cast<char*>(&sortingColumnCount), sizeof(int32_t)); // read number of sorting columns

            std::vector<std::string> columnNames;
            std::vector<std::string> sortingColumnNames;

            for (int32_t j = 0; j < sortingColumnCount; j++)
            {
                int32_t sortingColumnLength;
                dbFile.read(reinterpret_cast<char*>(&sortingColumnLength), sizeof(int32_t)); // read sorting column name length

                std::unique_ptr<char[]> sortingColumnName(new char[sortingColumnLength]);
                dbFile.read(sortingColumnName.get(), sortingColumnLength); // read sorting column name
                BOOST_LOG_TRIVIAL(debug) << "Sorting column (table: " + std::string(tableName.get()) +
                                                ") loaded: " + std::string(sortingColumnName.get()) + ".";
                sortingColumnNames.push_back(sortingColumnName.get());
            }

            for (int32_t j = 0; j < columnCount; j++)
            {
                int32_t columnNameLength;
                dbFile.read(reinterpret_cast<char*>(&columnNameLength), sizeof(int32_t)); // read column name length

                std::unique_ptr<char[]> columnName(new char[columnNameLength]);
                dbFile.read(columnName.get(), columnNameLength); // read column name

                Context::getInstance().CheckColumnsLimit(columnNames.size());

                columnNames.push_back(columnName.get());
            }

            auto& table = database->tables_.at(tableName.get());
            table.SetSortingColumns(sortingColumnNames);

            std::vector<std::thread> threads;

            for (const std::string& columnName : columnNames)
            {
                threads.emplace_back(Database::LoadColumn, path, dbName.get(),
                                     persistenceFormatVersion, std::ref(table), std::ref(columnName));
            }

            for (int i = 0; i < columnNames.size(); i++)
            {
                threads[i].join();
            }
        }

        dbFile.close();

        return database;
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "File " + filePath + " is empty and so cannot be loaded.";
        return nullptr;
    }
}

/// <summary>
/// Load column of a table into memory from disk.
/// </summary>
/// <param name="path">Path directory, where column file (*.col) is.</param>
/// <param name="dbName">Name of the database.</param>
/// <param name="persistenceFormatVersion">Version of format used to persist .db and .col files into
/// disk.</param> <param name="table">Instance of table into which the column should be
/// added.</param> <param name="columnName">Names of particular column.</param>
void Database::LoadColumn(const char* path,
                          const char* dbName,
                          const int32_t persistenceFormatVersion,
                          Table& table,
                          const std::string& columnName)
{
    const int32_t oneChunkSize = 8 * 1024 * 1024;
    // read files .col:
    const std::string filePath = std::string(path) + std::string(dbName) + SEPARATOR +
                                 table.GetName() + SEPARATOR + columnName + ".col";

    std::ifstream colFile(filePath, std::ios::binary);

    colFile.seekg(0, colFile.end);
    const size_t fileSize = colFile.tellg();
    if (fileSize != 0)
    {
        colFile.seekg(0, colFile.beg);
        BOOST_LOG_TRIVIAL(info) << "Loading .col file with name: " << filePath << ".";

        int32_t emptyBlockIndex = 0;

        int32_t type;
        bool isNullable;
        bool isUnique;

        colFile.read(reinterpret_cast<char*>(&type), sizeof(int32_t)); // read type of column
        colFile.read(reinterpret_cast<char*>(&isNullable), sizeof(bool)); // read nullability of column
        colFile.read(reinterpret_cast<char*>(&isUnique), sizeof(bool)); // read unicity of column

        int32_t nullBitMaskAllocationSize = NullValues::GetNullBitMaskSize(table.GetBlockSize());

        switch (type)
        {
        case COLUMN_POLYGON:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_POLYGON, isNullable, isUnique);

            auto& columnPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(
                *table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int64_t)); // read data length (data block length)

                if (index != emptyBlockIndex) // there is null block
                {
                    columnPolygon.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug) << "Added empty ComplexPolygon block (" + filePath + ") at index: "
                                             << emptyBlockIndex;
                }
                else // read data from block
                {
                    auto& block = columnPolygon.AddBlock(groupId);
                    int64_t byteIndex = 0;
                    int32_t dataCount = 0;

                    int64_t remainingDataLength = dataLength;
                    while (byteIndex < dataLength)
                    {
                        int32_t currentChunkSize =
                            oneChunkSize < remainingDataLength ? oneChunkSize : remainingDataLength;

                        std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
                        std::unique_ptr<char[]> data(new char[currentChunkSize]);

                        colFile.read(data.get(), currentChunkSize);

                        int32_t byteIndexForChunks = 0;
                        int32_t remainingChunkSize = currentChunkSize;
                        int32_t dataCountInOneChunk = 0;
                        while (byteIndexForChunks < currentChunkSize)
                        {
                            int32_t entryByteLength = 0;
                            if (byteIndexForChunks + sizeof(int32_t) - 1 >= currentChunkSize)
                            {
                                int32_t restSize = (byteIndexForChunks + sizeof(int32_t) - currentChunkSize);
                                currentChunkSize += restSize;
                                std::unique_ptr<char[]> dataRest(new char[restSize]);
                                colFile.read(dataRest.get(), restSize);
                                memcpy(&entryByteLength, &data[byteIndexForChunks], sizeof(int32_t) - restSize);
                                memcpy(reinterpret_cast<char*>(&entryByteLength) + sizeof(int32_t) - restSize,
                                       dataRest.get(), restSize);
                            }
                            else
                            {
                                entryByteLength = *reinterpret_cast<int32_t*>(&data[byteIndexForChunks]);
                            }
                            std::unique_ptr<char[]> byteArray(new char[entryByteLength]);


                            byteIndex += sizeof(int32_t);
                            byteIndexForChunks += sizeof(int32_t);

                            if ((currentChunkSize - byteIndexForChunks) < entryByteLength)
                            {
                                int32_t dataLeftInCurrentChunk = currentChunkSize - byteIndexForChunks > 0 ?
                                                                     currentChunkSize - byteIndexForChunks :
                                                                     0;
                                std::unique_ptr<char[]> dataRest(new char[entryByteLength - dataLeftInCurrentChunk]);
                                currentChunkSize += entryByteLength - dataLeftInCurrentChunk;
                                colFile.read(dataRest.get(), entryByteLength - dataLeftInCurrentChunk);

                                if (dataLeftInCurrentChunk > 0)
                                {
                                    memcpy(byteArray.get(), &data[byteIndexForChunks], dataLeftInCurrentChunk);
                                }
                                memcpy(byteArray.get() + dataLeftInCurrentChunk, &dataRest[0],
                                       entryByteLength - dataLeftInCurrentChunk);


                                ColmnarDB::Types::ComplexPolygon entryDataPolygon;
                                entryDataPolygon.ParseFromArray(byteArray.get(), entryByteLength);
                                dataPolygon.push_back(entryDataPolygon);

                                byteIndexForChunks += entryByteLength;
                                remainingChunkSize = 0;
                            }
                            else
                            {
                                memcpy(byteArray.get(), &data[byteIndexForChunks], entryByteLength);
                                remainingChunkSize -= entryByteLength;

                                ColmnarDB::Types::ComplexPolygon entryDataPolygon;
                                entryDataPolygon.ParseFromArray(byteArray.get(), entryByteLength);
                                dataPolygon.push_back(entryDataPolygon);

                                byteIndexForChunks += entryByteLength;
                            }

                            dataCountInOneChunk++;
                            dataCount++;
                            byteIndex += entryByteLength;

                            if (dataCount > columnPolygon.GetBlockSize())
                            {
                                throw std::runtime_error("Loaded data (" + filePath + ") from disk does not fit into existing block");
                                break;
                            }
                        }
                        remainingDataLength -= currentChunkSize;

                        if (isUnique)
                        {
                            if (isNullable)
                            {
                                throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                            }

                            for (int32_t i = 0; i < dataPolygon.size(); i++)
                            {
                                if (!columnPolygon.IsDuplicate(dataPolygon[i]))
                                {
                                    columnPolygon.InsertIntoHashmap(dataPolygon[i]);
                                }
                                else
                                {
                                    throw std::runtime_error(
                                        "Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                        ComplexPolygonFactory::WktFromPolygon(dataPolygon[i]));
                                }
                            }
                        }

                        block.InsertData(dataPolygon);
                    }

                    block.SetNullBitmask(std::move(nullBitMask));
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added ComplexPolygon block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_POINT:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_POINT, isNullable, isUnique);

            auto& columnPoint =
                dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;
                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(int64_t)); // read byte data length (data block length)

                if (index != emptyBlockIndex) // there is null block
                {
                    columnPoint.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Point block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    auto& block = columnPoint.AddBlock(groupId);
                    int64_t byteIndex = 0;
                    int32_t dataCount = 0;

                    int64_t remainingDataLength = dataLength;

                    while (byteIndex < dataLength)
                    {
                        int32_t currentChunkSize =
                            oneChunkSize < remainingDataLength ? oneChunkSize : remainingDataLength;

                        std::vector<ColmnarDB::Types::Point> dataPoint;
                        std::unique_ptr<char[]> data(new char[currentChunkSize]);

                        colFile.read(data.get(), currentChunkSize);

                        int32_t byteIndexForChunks = 0;
                        int32_t remainingChunkSize = currentChunkSize;
                        int32_t dataCountInOneChunk = 0;
                        while (byteIndexForChunks < currentChunkSize)
                        {
                            int32_t entryByteLength = 0;
                            if (byteIndexForChunks + sizeof(int32_t) - 1 >= currentChunkSize)
                            {
                                int32_t restSize = (byteIndexForChunks + sizeof(int32_t) - currentChunkSize);
                                currentChunkSize += restSize;
                                std::unique_ptr<char[]> dataRest(new char[restSize]);
                                colFile.read(dataRest.get(), restSize);
                                memcpy(&entryByteLength, &data[byteIndexForChunks], sizeof(int32_t) - restSize);
                                memcpy(reinterpret_cast<char*>(&entryByteLength) + sizeof(int32_t) - restSize,
                                       dataRest.get(), restSize);
                            }
                            else
                            {
                                entryByteLength = *reinterpret_cast<int32_t*>(&data[byteIndexForChunks]);
                            }
                            std::unique_ptr<char[]> byteArray(new char[entryByteLength]);

                            byteIndex += sizeof(int32_t);
                            byteIndexForChunks += sizeof(int32_t);

                            if ((currentChunkSize - byteIndexForChunks) < entryByteLength)
                            {
                                int32_t dataLeftInCurrentChunk = currentChunkSize - byteIndexForChunks > 0 ?
                                                                     currentChunkSize - byteIndexForChunks :
                                                                     0;
                                std::unique_ptr<char[]> dataRest(new char[entryByteLength - dataLeftInCurrentChunk]);
                                currentChunkSize += entryByteLength - dataLeftInCurrentChunk;
                                colFile.read(dataRest.get(), entryByteLength - dataLeftInCurrentChunk);

                                colFile.read(dataRest.get(), entryByteLength - dataLeftInCurrentChunk);
                                if (dataLeftInCurrentChunk > 0)
                                {
                                    memcpy(byteArray.get(), &data[byteIndexForChunks], dataLeftInCurrentChunk);
                                }
                                memcpy(byteArray.get() + dataLeftInCurrentChunk, &dataRest[0],
                                       entryByteLength - dataLeftInCurrentChunk);

                                ColmnarDB::Types::Point entryDataPoint;
                                entryDataPoint.ParseFromArray(byteArray.get(), entryByteLength);
                                dataPoint.push_back(entryDataPoint);

                                byteIndexForChunks += entryByteLength;
                                remainingChunkSize = 0;
                            }
                            else
                            {
                                memcpy(byteArray.get(), &data[byteIndexForChunks], entryByteLength);
                                remainingChunkSize -= entryByteLength;

                                ColmnarDB::Types::Point entryDataPoint;
                                entryDataPoint.ParseFromArray(byteArray.get(), entryByteLength);
                                dataPoint.push_back(entryDataPoint);

                                byteIndexForChunks += entryByteLength;
                            }

                            dataCountInOneChunk++;
                            dataCount++;
                            byteIndex += entryByteLength;

                            if (dataCount > columnPoint.GetBlockSize())
                            {
                                throw std::runtime_error("Loaded data (" + filePath + ") from disk does not fit into existing block");
                                break;
                            }
                        }
                        remainingDataLength -= currentChunkSize;

                        if (isUnique)
                        {
                            if (isNullable)
                            {
                                throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                            }

                            for (int32_t i = 0; i < dataPoint.size(); i++)
                            {
                                if (!columnPoint.IsDuplicate(dataPoint[i]))
                                {
                                    columnPoint.InsertIntoHashmap(dataPoint[i]);
                                }
                                else
                                {
                                    throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                                             PointFactory::WktFromPoint(dataPoint[i]));
                                }
                            }
                        }
                        block.InsertData(dataPoint);
                    }

                    block.SetNullBitmask(std::move(nullBitMask));
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Point block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_STRING:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_STRING, isNullable, isUnique);

            auto& columnString = dynamic_cast<ColumnBase<std::string>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int64_t)); // read data length (data block length)

                if (index != emptyBlockIndex) // there is null block
                {
                    columnString.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty String block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    auto& block = columnString.AddBlock(groupId);
                    int64_t byteIndex = 0;
                    int32_t dataCount = 0;

                    int64_t remainingDataLength = dataLength;
                    while (byteIndex < dataLength)
                    {
                        int32_t currentChunkSize =
                            oneChunkSize < remainingDataLength ? oneChunkSize : remainingDataLength;

                        std::vector<std::string> dataString;
                        std::unique_ptr<char[]> data(new char[currentChunkSize]);

                        colFile.read(data.get(), currentChunkSize);

                        int32_t byteIndexForChunks = 0;
                        int32_t remainingChunkSize = currentChunkSize;
                        int32_t dataCountInOneChunk = 0;
                        while (byteIndexForChunks < currentChunkSize)
                        {
                            int32_t entryByteLength = 0;
                            if (byteIndexForChunks + sizeof(int32_t) - 1 >= currentChunkSize)
                            {
                                int32_t restSize = (byteIndexForChunks + sizeof(int32_t) - currentChunkSize);
                                currentChunkSize += restSize;
                                std::unique_ptr<char[]> dataRest(new char[restSize]);
                                colFile.read(dataRest.get(), restSize);
                                memcpy(&entryByteLength, &data[byteIndexForChunks], sizeof(int32_t) - restSize);
                                memcpy(reinterpret_cast<char*>(&entryByteLength) + sizeof(int32_t) - restSize,
                                       dataRest.get(), restSize);
                            }
                            else
                            {
                                entryByteLength = *reinterpret_cast<int32_t*>(&data[byteIndexForChunks]);
                            }
                            std::unique_ptr<char[]> byteArray(new char[entryByteLength]);


                            byteIndex += sizeof(int32_t);
                            byteIndexForChunks += sizeof(int32_t);


                            if ((currentChunkSize - byteIndexForChunks) < entryByteLength)
                            {
                                int32_t dataLeftInCurrentChunk = currentChunkSize - byteIndexForChunks > 0 ?
                                                                     currentChunkSize - byteIndexForChunks :
                                                                     0;
                                std::unique_ptr<char[]> dataRest(new char[entryByteLength - dataLeftInCurrentChunk]);
                                currentChunkSize += entryByteLength - dataLeftInCurrentChunk;

                                colFile.read(dataRest.get(), entryByteLength - dataLeftInCurrentChunk);
                                if (dataLeftInCurrentChunk > 0)
                                {
                                    memcpy(byteArray.get(), &data[byteIndexForChunks], dataLeftInCurrentChunk);
                                }
                                memcpy(byteArray.get() + dataLeftInCurrentChunk, &dataRest[0],
                                       entryByteLength - dataLeftInCurrentChunk);


                                std::string entryDataString(byteArray.get());
                                dataString.push_back(entryDataString);


                                byteIndexForChunks += entryByteLength;
                                remainingChunkSize = 0;
                            }
                            else
                            {
                                memcpy(byteArray.get(), &data[byteIndexForChunks], entryByteLength);
                                remainingChunkSize -= entryByteLength;


                                std::string entryDataString(byteArray.get());
                                dataString.push_back(entryDataString);


                                byteIndexForChunks += entryByteLength;
                            }


                            dataCountInOneChunk++;
                            dataCount++;
                            byteIndex += entryByteLength;


                            if (dataCount > columnString.GetBlockSize())
                            {
                                throw std::runtime_error("Loaded data (" + filePath + ") from disk does not fit into existing block");
                                break;
                            }
                        }
                        remainingDataLength -= currentChunkSize;

                        if (isUnique)
                        {
                            if (isNullable)
                            {
                                throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                            }

                            for (int32_t i = 0; i < dataString.size(); i++)
                            {
                                if (!columnString.IsDuplicate(dataString[i]))
                                {
                                    columnString.InsertIntoHashmap(dataString[i]);
                                }
                                else
                                {
                                    throw std::runtime_error(
                                        "Loaded column: " + filePath +
                                        " has UNIQUE constraint and duplicate values: " + dataString[i]);
                                }
                            }
                        }
                        block.InsertData(dataString);
                    }

                    block.SetNullBitmask(std::move(nullBitMask));
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added String block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_INT8_T:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_INT8_T, isNullable, isUnique);

            auto& columnInt = dynamic_cast<ColumnBase<int8_t>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int32_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); // read data length (number of entries)
                int8_t isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // read whether compressed
                int8_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int8_t)); // read statistics min
                int8_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int8_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int8_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int8_t)); // read statistics sum

                if (index != emptyBlockIndex) // there is null block
                {
                    columnInt.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Int8 block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    std::unique_ptr<int8_t[]> data = nullptr;
                    data = std::unique_ptr<int8_t[]>(new int8_t[columnInt.GetBlockSize()]);

                    colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int8_t)); // read entry data

                    if (dataLength > columnInt.GetBlockSize())
                    {
                        throw std::runtime_error("Loaded data (" + filePath +
                                                 ") from disk does not fit into existing block");
                        break;
                    }

                    if (isUnique)
                    {
                        if (isNullable)
                        {
                            throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                        }
                        std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                      [&columnInt, &filePath](int8_t& value) {
                                          if (!columnInt.IsDuplicate(value))
                                          {
                                              columnInt.InsertIntoHashmap(value);
                                          }
                                          else
                                          {
                                              throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                                                       std::to_string(value));
                                          }
                                      });
                    }

                    auto& block =
                        columnInt.AddBlock(std::move(data), dataLength, columnInt.GetBlockSize(),
                                           groupId, false, static_cast<bool>(isCompressed), false);
                    block.SetNullBitmask(std::move(nullBitMask));
                    block.setBlockStatistics(min, max, avg, sum, dataLength);

                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Int8 block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_INT:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_INT, isNullable, isUnique);

            auto& columnInt = dynamic_cast<ColumnBase<int32_t>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int32_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); // read data length (number of entries)
                int8_t isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // read whether compressed
                int32_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int32_t)); // read statistics min
                int32_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int32_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int32_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int32_t)); // read statistics sum

                if (index != emptyBlockIndex) // there is null block
                {
                    columnInt.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Int32 block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    std::unique_ptr<int32_t[]> data = nullptr;
                    data = std::unique_ptr<int32_t[]>(new int32_t[columnInt.GetBlockSize()]);

                    colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int32_t)); // read entry data

                    if (dataLength > columnInt.GetBlockSize())
                    {
                        throw std::runtime_error("Loaded data (" + filePath +
                                                 ") from disk does not fit into existing block");
                        break;
                    }

                    if (isUnique)
                    {
                        if (isNullable)
                        {
                            throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                        }
                        std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                      [&columnInt, &filePath](int32_t& value) {
                                          if (!columnInt.IsDuplicate(value))
                                          {
                                              columnInt.InsertIntoHashmap(value);
                                          }
                                          else
                                          {
                                              throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                                                       std::to_string(value));
                                          }
                                      });
                    }

                    auto& block =
                        columnInt.AddBlock(std::move(data), dataLength, columnInt.GetBlockSize(),
                                           groupId, false, static_cast<bool>(isCompressed), false);
                    block.SetNullBitmask(std::move(nullBitMask));
                    block.setBlockStatistics(min, max, avg, sum, dataLength);

                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Int32 block (" + filePath + ") with data at index : " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_LONG:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_LONG, isNullable, isUnique);

            auto& columnLong = dynamic_cast<ColumnBase<int64_t>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int32_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); // read data length (number of entries)
                int8_t isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // read whether compressed
                int64_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int64_t)); // read statistics min
                int64_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int64_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int64_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int64_t)); // read statistics sum

                if (index != emptyBlockIndex) // there is null block
                {
                    columnLong.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Int64 block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    std::unique_ptr<int64_t[]> data = nullptr;
                    data = std::unique_ptr<int64_t[]>(new int64_t[columnLong.GetBlockSize()]);

                    colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int64_t)); // read entry data

                    if (dataLength > columnLong.GetBlockSize())
                    {
                        throw std::runtime_error("Loaded data (" + filePath +
                                                 ") from disk does not fit into existing block");
                        break;
                    }

                    if (isUnique)
                    {
                        if (isNullable)
                        {
                            throw std::runtime_error("Loaded column: " + columnName + " has UNIQUE constraint and has not NOT NULL constraint");
                        }
                        std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                      [&columnLong, &columnName](int64_t& value) {
                                          if (!columnLong.IsDuplicate(value))
                                          {
                                              columnLong.InsertIntoHashmap(value);
                                          }
                                          else
                                          {
                                              throw std::runtime_error("Loaded column: " + columnName + " has UNIQUE constraint and duplicate values: " +
                                                                       std::to_string(value));
                                          }
                                      });
                    }

                    auto& block =
                        columnLong.AddBlock(std::move(data), dataLength, columnLong.GetBlockSize(),
                                            groupId, false, static_cast<bool>(isCompressed), false);
                    block.SetNullBitmask(std::move(nullBitMask));
                    block.setBlockStatistics(min, max, avg, sum, dataLength);

                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Int64 block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_FLOAT:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_FLOAT, isNullable, isUnique);

            auto& columnFloat = dynamic_cast<ColumnBase<float>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int32_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); // read data length (number of entries)
                int8_t isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // read whether compressed
                float min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(float)); // read statistics min
                float max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(float)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                float sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(float)); // read statistics sum

                if (index != emptyBlockIndex) // there is null block
                {
                    columnFloat.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Float block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    std::unique_ptr<float[]> data = nullptr;
                    data = std::unique_ptr<float[]>(new float[columnFloat.GetBlockSize()]);

                    colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(float)); // read entry data

                    if (dataLength > columnFloat.GetBlockSize())
                    {
                        throw std::runtime_error("Loaded data (" + filePath +
                                                 ") from disk does not fit into existing block");
                        break;
                    }

                    if (isUnique)
                    {
                        if (isNullable)
                        {
                            throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                        }
                        std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                      [&columnFloat, &filePath](float& value) {
                                          if (!columnFloat.IsDuplicate(value))
                                          {
                                              columnFloat.InsertIntoHashmap(value);
                                          }
                                          else
                                          {
                                              throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                                                       std::to_string(value));
                                          }
                                      });
                    }

                    auto& block =
                        columnFloat.AddBlock(std::move(data), dataLength, columnFloat.GetBlockSize(),
                                             groupId, false, static_cast<bool>(isCompressed), false);
                    block.SetNullBitmask(std::move(nullBitMask));
                    block.setBlockStatistics(min, max, avg, sum, dataLength);

                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Float block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        case COLUMN_DOUBLE:
        {
            table.CreateColumn(columnName.c_str(), COLUMN_DOUBLE, isNullable, isUnique);

            auto& columnDouble = dynamic_cast<ColumnBase<double>&>(*table.GetColumns().at(columnName));

            while (!colFile.eof())
            {
                int32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<nullmask_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<nullmask_t[]>(new nullmask_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << filePath << " has finished successfully.";
                    break;
                }

                int32_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); // read data length (number of entries)
                int8_t isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // read whether compressed
                double min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(double)); // read statistics min
                double max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(double)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                double sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(double)); // read statistics sum

                if (index != emptyBlockIndex) // there is null block
                {
                    columnDouble.AddBlock(); // add empty block
                    BOOST_LOG_TRIVIAL(debug)
                        << "Added empty Double block (" + filePath + ") at index: " << emptyBlockIndex;
                }
                else // read data from block
                {
                    std::unique_ptr<double[]> data = nullptr;
                    data = std::unique_ptr<double[]>(new double[columnDouble.GetBlockSize()]);

                    colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(double)); // read entry data

                    if (dataLength > columnDouble.GetBlockSize())
                    {
                        throw std::runtime_error("Loaded data (" + filePath +
                                                 ") from disk does not fit into existing block");
                        break;
                    }

                    if (isUnique)
                    {
                        if (isNullable)
                        {
                            throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and has not NOT NULL constraint");
                        }
                        std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                      [&columnDouble, &filePath](double& value) {
                                          if (!columnDouble.IsDuplicate(value))
                                          {
                                              columnDouble.InsertIntoHashmap(value);
                                          }
                                          else
                                          {
                                              throw std::runtime_error("Loaded column: " + filePath + " has UNIQUE constraint and duplicate values: " +
                                                                       std::to_string(value));
                                          }
                                      });
                    }

                    auto& block = columnDouble.AddBlock(std::move(data), dataLength,
                                                        columnDouble.GetBlockSize(), groupId, false,
                                                        static_cast<bool>(isCompressed), false);
                    block.SetNullBitmask(std::move(nullBitMask));
                    block.setBlockStatistics(min, max, avg, sum, dataLength);

                    BOOST_LOG_TRIVIAL(debug)
                        << "Added Double block (" + filePath + ") with data at index: " << index;
                }

                emptyBlockIndex += 1;
            }
        }
        break;

        default:
            BOOST_LOG_TRIVIAL(error) << "Unsupported data type (when loading database - "
                                     << std::string(path) << std::string(dbName) << "): " << type;
            throw std::domain_error("Unsupported data type (when loading database - " + std::string(path) +
                                    std::string(dbName) + "): " + std::to_string(type));
        }

        table.GetColumns().at(columnName)->UpdateSize(); // Column with special type needs to recount size
        colFile.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "File " + filePath + " is empty and so cannot be loaded.";
    }
}

/// <summary>
/// Creates table with given name and columns and adds it to database. If the table already
/// existed, create missing columns if there are any missing
/// </summary>
/// <param name="columns">Columns with types.</param>
/// <param name="tableName">Table name.</param>
/// <param name="areNullable">Nullablity of columns. Default values are set to be true.</param>
/// <param name="blockSize">Table block size. If not specified, as the default value a database
/// block size will be used.</param>
/// <returns>Newly created table.</returns>
Table& Database::CreateTable(const std::unordered_map<std::string, DataType>& columns,
                             const char* tableName,
                             const std::unordered_map<std::string, bool>& areNullable,
                             const std::unordered_map<std::string, bool>& areUnique,
                             int32_t blockSize)
{
    Context::getInstance().CheckTablesLimit(tables_.size());
    Context::getInstance().CheckColumnsLimit(columns.size() - 1);

    auto search = tables_.find(tableName);

    if (search != tables_.end())
    {
        auto& table = search->second;

        for (const auto& entry : columns)
        {
            if (table.ContainsColumn(entry.first.c_str()))
            {
                auto& tableColumns = table.GetColumns();

                if (tableColumns.at(entry.first)->GetColumnType() != entry.second)
                {
                    throw std::domain_error(
                        "Column type in CreateTable does not match with existing column.");
                }
            }
            else
            {
                bool isNullable = areNullable.empty() ? true : areNullable.at(entry.first);
                bool isUnique = areUnique.empty() ? false : areUnique.at(entry.first);
                table.CreateColumn(entry.first.c_str(), entry.second, isNullable, isUnique);
            }
        }

        return table;
    }
    else
    {
        if (blockSize == -1)
        {
            // if table block size was not specified, use as the default value the block size from database
            tables_.emplace(std::make_pair(tableName, Table(Database::GetDatabaseByName(name_),
                                                            tableName, blockSize_)));
        }
        else
        {
            // if table block size was specified, use it as table block size for this particular table
            tables_.emplace(std::make_pair(tableName, Table(Database::GetDatabaseByName(name_),
                                                            tableName, blockSize)));
        }

        auto& table = tables_.at(tableName);

        for (auto& entry : columns)
        {
            bool isNullable = areNullable.empty() ? true : areNullable.at(entry.first);
            bool isUnique = areUnique.empty() ? false : areUnique.at(entry.first);
            table.CreateColumn(entry.first.c_str(), entry.second, isNullable, isUnique);
        }

        return table;
    }
}

/// <summary>
/// Add database to in memory list.
/// </summary>
/// <param name="database">Database to be added.</param>
void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database> database)
{
    std::lock_guard<std::mutex> lock(dbAccessMutex_);
    Context::getInstance().CheckDatabasesLimit(Context::getInstance().GetLoadedDatabases().size());

    if (!Context::getInstance().GetLoadedDatabases().insert({database->name_, database}).second)
    {
        throw std::invalid_argument("Attempt to insert duplicate database name");
    }
}

/// <summary>
/// Remove database from in memory database list.
/// </summary>
/// <param name="databaseName">Name of database to be removed.</param>
void Database::RemoveFromInMemoryDatabaseList(const char* databaseName)
{
    // erase db from map
    std::lock_guard<std::mutex> lock(dbAccessMutex_);
    Context::getInstance().GetLoadedDatabases().erase(databaseName);
}

/// <summary>
/// Write column into disk.
/// </summary>
/// <param name="column">Column to be written.</param>
/// <param name="pathStr">Path to database storage directory.</param>
/// <param name="name">Names of particular column.</param>
/// <param name="table">Names of particular table.</param>
void Database::WriteColumn(const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                           std::string pathStr,
                           std::string name,
                           const std::pair<const std::string, Table>& table)
{
    BOOST_LOG_TRIVIAL(debug) << "Saving .col file with name: " << pathStr << name << SEPARATOR
                             << table.first << SEPARATOR << column.second->GetName() << ".col";

    std::ofstream colFile(pathStr + "/" + name + SEPARATOR + table.first + SEPARATOR +
                              column.second->GetName() + ".col",
                          std::ios::binary);

    if (colFile.is_open())
    {
        int32_t type = column.second->GetColumnType();
        bool isNullable = column.second->GetIsNullable();
        bool isUnique = column.second->GetIsUnique();

        colFile.write(reinterpret_cast<char*>(&type), sizeof(int32_t)); // write type of column
        colFile.write(reinterpret_cast<char*>(&isNullable), sizeof(bool)); // write nullability of column
        colFile.write(reinterpret_cast<char*>(&isUnique), sizeof(bool)); // write unicity of column

        switch (type)
        {
        case COLUMN_POLYGON:
        {
            int32_t index = 0;

            const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

            for (const auto& block : colPolygon.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of ComplexPolygon data with index = " << index;

                auto data = block->GetData();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int64_t dataByteSize = 0;

                for (int32_t i = 0; i < dataLength; i++)
                {
                    dataByteSize += data[i].ByteSize();
                }

                int64_t dataRawLength = dataByteSize + dataLength * sizeof(int32_t);

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataRawLength), sizeof(int64_t)); // write block length in bytes
                    for (size_t i = 0; i < dataLength; i++)
                    {
                        int32_t entryByteLength = data[i].ByteSize();
                        std::unique_ptr<char[]> byteArray(new char[entryByteLength]);

                        data[i].SerializeToArray(byteArray.get(), entryByteLength);

                        colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); // write entry length
                        colFile.write(byteArray.get(), entryByteLength); // write entry data
                    }
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_POINT:
        {
            int32_t index = 0;

            const ColumnBase<ColmnarDB::Types::Point>& colPoint =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

            for (const auto& block : colPoint.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Point data with index = " << index;

                auto data = block->GetData();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int64_t dataByteSize = 0;

                for (int32_t i = 0; i < dataLength; i++)
                {
                    dataByteSize += data[i].ByteSize();
                }

                int64_t dataRawLength = dataByteSize + dataLength * sizeof(int32_t);

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataRawLength), sizeof(int64_t)); // write block length in bytes
                    for (size_t i = 0; i < dataLength; i++)
                    {
                        int32_t entryByteLength = data[i].ByteSize();
                        std::unique_ptr<char[]> byteArray(new char[entryByteLength]);

                        data[i].SerializeToArray(byteArray.get(), entryByteLength);

                        colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); // write entry length
                        colFile.write(byteArray.get(), entryByteLength); // write entry data
                    }
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_STRING:
        {
            int32_t index = 0;

            const ColumnBase<std::string>& colStr =
                dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

            for (const auto& block : colStr.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of String data with index = " << index;

                auto data = block->GetData();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int64_t dataByteSize = 0;

                for (int32_t i = 0; i < dataLength; i++)
                {
                    dataByteSize += data[i].length() + 1;
                }

                int64_t dataRawLength = dataByteSize + dataLength * sizeof(int32_t);

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataRawLength), sizeof(int64_t)); // write block length in bytes
                    for (size_t i = 0; i < dataLength; i++)
                    {
                        int32_t entryByteLength = data[i].length() + 1; // +1 because '\0'

                        colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); // write entry length
                        colFile.write(data[i].c_str(), entryByteLength); // write entry data
                    }
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_INT8_T:
        {
            int32_t index = 0;

            const ColumnBase<int8_t>& colInt = dynamic_cast<const ColumnBase<int8_t>&>(*(column.second));

            for (const auto& block : colInt.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Int8 data with index = " << index;

                auto data = block->GetData();
                int8_t isCompressed = (int8_t)block->IsCompressed();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int8_t min = block->GetMin();
                int8_t max = block->GetMax();
                float avg = block->GetAvg();
                int8_t sum = block->GetSum();

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataLength),
                                  sizeof(int32_t)); // write block length (number of entries)
                    colFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // write whether compressed
                    colFile.write(reinterpret_cast<char*>(&min), sizeof(int8_t)); // write statistics min
                    colFile.write(reinterpret_cast<char*>(&max), sizeof(int8_t)); // write statistics max
                    colFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colFile.write(reinterpret_cast<char*>(&sum), sizeof(int8_t)); // write statistics sum
                    colFile.write(reinterpret_cast<const char*>(data), dataLength * sizeof(int8_t)); // write block of data
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_INT:
        {
            int32_t index = 0;

            const ColumnBase<int32_t>& colInt = dynamic_cast<const ColumnBase<int32_t>&>(*(column.second));

            for (const auto& block : colInt.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Int32 data with index = " << index;

                auto data = block->GetData();
                int8_t isCompressed = (int8_t)block->IsCompressed();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int32_t min = block->GetMin();
                int32_t max = block->GetMax();
                float avg = block->GetAvg();
                int32_t sum = block->GetSum();

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataLength),
                                  sizeof(int32_t)); // write block length (number of entries)
                    colFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // write whether compressed
                    colFile.write(reinterpret_cast<char*>(&min), sizeof(int32_t)); // write statistics min
                    colFile.write(reinterpret_cast<char*>(&max), sizeof(int32_t)); // write statistics max
                    colFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colFile.write(reinterpret_cast<char*>(&sum), sizeof(int32_t)); // write statistics sum
                    colFile.write(reinterpret_cast<const char*>(data), dataLength * sizeof(int32_t)); // write block of data
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_LONG:
        {
            int32_t index = 0;

            const ColumnBase<int64_t>& colLong = dynamic_cast<const ColumnBase<int64_t>&>(*(column.second));

            for (const auto& block : colLong.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Int64 data with index = " << index;

                auto data = block->GetData();
                int8_t isCompressed = (int8_t)block->IsCompressed();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                int64_t min = block->GetMin();
                int64_t max = block->GetMax();
                float avg = block->GetAvg();
                int64_t sum = block->GetSum();

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataLength),
                                  sizeof(int32_t)); // write block length (number of entries)
                    colFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // write whether compressed
                    colFile.write(reinterpret_cast<char*>(&min), sizeof(int64_t)); // write statistics min
                    colFile.write(reinterpret_cast<char*>(&max), sizeof(int64_t)); // write statistics max
                    colFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colFile.write(reinterpret_cast<char*>(&sum), sizeof(int64_t)); // write statistics sum
                    colFile.write(reinterpret_cast<const char*>(data), dataLength * sizeof(int64_t)); // write block of data
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_FLOAT:
        {
            int32_t index = 0;

            const ColumnBase<float>& colFloat = dynamic_cast<const ColumnBase<float>&>(*(column.second));

            for (const auto& block : colFloat.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Float data with index = " << index;

                auto data = block->GetData();
                int8_t isCompressed = (int8_t)block->IsCompressed();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                float min = block->GetMin();
                float max = block->GetMax();
                float avg = block->GetAvg();
                float sum = block->GetSum();

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataLength),
                                  sizeof(int32_t)); // write block length (number of entries)
                    colFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // write whether compressed
                    colFile.write(reinterpret_cast<char*>(&min), sizeof(float)); // write statistics min
                    colFile.write(reinterpret_cast<char*>(&max), sizeof(float)); // write statistics max
                    colFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colFile.write(reinterpret_cast<char*>(&sum), sizeof(float)); // write statistics sum
                    colFile.write(reinterpret_cast<const char*>(data), dataLength * sizeof(float)); // write block of data
                    index += 1;
                }
            }
        }
        break;

        case COLUMN_DOUBLE:
        {
            int32_t index = 0;

            const ColumnBase<double>& colDouble = dynamic_cast<const ColumnBase<double>&>(*(column.second));

            for (const auto& block : colDouble.GetBlocksList())
            {
                BOOST_LOG_TRIVIAL(debug) << "Saving block of Double data with index = " << index;

                auto data = block->GetData();
                int8_t isCompressed = (int8_t)block->IsCompressed();
                int32_t groupId = block->GetGroupId();
                int32_t dataLength = block->GetSize();
                double min = block->GetMin();
                double max = block->GetMax();
                float avg = block->GetAvg();
                double sum = block->GetSum();

                if (dataLength > 0)
                {
                    colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); // write index
                    colFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength = NullValues::GetNullBitMaskSizeInBytes(block->GetSize());
                        colFile.write(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // write nullBitMask length
                        colFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                      nullBitMaskLength); // write nullBitMask
                    }
                    colFile.write(reinterpret_cast<char*>(&dataLength),
                                  sizeof(int32_t)); // write block length (number of entries)
                    colFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(int8_t)); // write whether compressed
                    colFile.write(reinterpret_cast<char*>(&min), sizeof(double)); // write statistics min
                    colFile.write(reinterpret_cast<char*>(&max), sizeof(double)); // write statistics max
                    colFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colFile.write(reinterpret_cast<char*>(&sum), sizeof(double)); // write statistics sum
                    colFile.write(reinterpret_cast<const char*>(data), dataLength * sizeof(double)); // write block of data
                    index += 1;
                }
            }
        }
        break;

        default:
            throw std::domain_error("Unsupported data type (when persisting database): " + std::to_string(type));
            break;
        }

        colFile.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(error)
            << "Could not open file " +
                   std::string(pathStr + "/" + name + SEPARATOR + table.first + SEPARATOR +
                               column.second->GetName() + ".col") +
                   " for writing. Persisting .col file was not successful. Check if the process "
                   "have write access into the folder or file.";
    }
}
