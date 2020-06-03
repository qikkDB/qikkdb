#include <boost/filesystem.hpp>
#include <boost\algorithm\string\case_conv.hpp>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>

#include "json/json.h"
#include "Configuration.h"
#include "Database.h"

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
/// <param name="columnName">Name of the column which data will be copied.</param>
void Database::CopyBlocksOfColumn(Table& srcTable, Table& dstTable, const std::string& columnName)
{
    BOOST_LOG_TRIVIAL(debug) << "Database: Copying data (column name: " << columnName
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
    }

    BOOST_LOG_TRIVIAL(debug) << "Database: Copying data (column name: " << columnName
                             << ") from the table named: " << srcTable.GetName()
                             << " to the table named: " << dstTable.GetName() << " has finished.";
}

/// <summary>
/// Save only DB_EXTENSION file to disk into directory defined in configuration file.
/// </summary>
void Database::PersistOnlyDbFile()
{
    const std::string path = Configuration::GetInstance().GetDatabaseDir();

    boost::filesystem::create_directories(path);

    const std::string filePath = std::string(path + name_ + DB_EXTENSION);

    // write file DB_EXTENSION
    BOOST_LOG_TRIVIAL(debug) << "Database: Saving " << DB_EXTENSION << " file with name : " << filePath;
    std::ofstream dbFile(filePath, std::ios::binary);

    if (dbFile.is_open())
    {
        Json::Value rootJSON;
        Json::Value tableArrayJSON(Json::arrayValue);
        Json::StreamWriterBuilder builder;
        const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

        rootJSON["persistence_format_version"] = PERSISTENCE_FORMAT_VERSION; // write persistence format version
        rootJSON["database_name"] = name_; // write db name
        rootJSON["database_default_block_size"] = blockSize_; // write block size
        for (auto& table : tables_)
        {
            const auto& columns = table.second.GetColumns();
            const auto& sortingColumns = table.second.GetSortingColumns();
            const std::string& tableName = table.second.GetName();

            Json::Value tableJSON;
            Json::Value indexColumnsArrayJSON(Json::arrayValue);
            Json::Value columnsArrayJSON(Json::arrayValue);

            tableJSON["table_name"] = tableName; // write table name
            tableJSON["table_block_size"] = table.second.GetBlockSize(); // write table block size
            tableJSON["start_up_loading"] = true;
            tableJSON["save_interval_ms"] = table.second.GetSaveInterval();

            if (sortingColumns.size() > 0)
            {
                for (const std::string sortingColumn : sortingColumns)
                {
                    Json::Value indexColumnsJSON;
                    indexColumnsJSON["index_column_name"] = sortingColumn; // write sorting column name
                    indexColumnsArrayJSON.append(indexColumnsJSON);
                }
            }

            tableJSON["index_columns"] = indexColumnsArrayJSON;

            for (const auto& column : columns)
            {
                const std::string columnName = column.first;
                const DataType columnType = column.second->GetColumnType();

                Json::Value columnsJSON;
                columnsJSON["column_name"] = columnName; // write column name
                columnsJSON["column_type"] = columnType; // write column type

                const std::string fileAddressPath = column.second->GetFileAddressPath();
                const std::string fileDataPath = column.second->GetFileDataPath();

                if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
                {
                    columnsJSON["file_path_address_file"] =
                        path + name_ + SEPARATOR + tableName + SEPARATOR + columnName + COLUMN_ADDRESS_EXTENSION;
                    BOOST_LOG_TRIVIAL(debug)
                        << "Database: Default address file path ( "
                        << path + name_ + SEPARATOR + tableName + SEPARATOR + columnName + COLUMN_ADDRESS_EXTENSION
                        << " ) has been persisted for column " << columnName << " of table "
                        << tableName << " of database " << name_ << ".";
                }
                else
                {
                    columnsJSON["file_path_address_file"] = fileAddressPath;
                    BOOST_LOG_TRIVIAL(debug) << "Database: Specific address file path ( " << fileAddressPath
                                             << " ) has been persisted for column " << columnName << " of table "
                                             << tableName << " of database " << name_ << ".";
                }

                if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
                {
                    columnsJSON["file_path_data_file"] = path + name_ + SEPARATOR + tableName +
                                                         SEPARATOR + columnName + COLUMN_DATA_EXTENSION;
                    BOOST_LOG_TRIVIAL(debug)
                        << "Database: Default data file path ( "
                        << path + name_ + SEPARATOR + tableName + SEPARATOR + columnName + COLUMN_DATA_EXTENSION
                        << " ) has been persisted for column " << columnName << " of table "
                        << tableName << " of database " << name_ << ".";
                }
                else
                {
                    columnsJSON["file_path_data_file"] = fileDataPath;
                    BOOST_LOG_TRIVIAL(debug) << "Database: Specific data file path ( " << fileDataPath
                                             << " ) has been persisted for column " << columnName << " of table "
                                             << tableName << " of database " << name_ << ".";
                }


                if (columnType == COLUMN_STRING || columnType == COLUMN_POLYGON)
                {
                    const std::string fileFragmentPath = column.second->GetFileFragmentPath();
                    const std::string encoding = column.second->GetEncoding();

                    if (fileFragmentPath.size() == 0 ||
                        fileFragmentPath == Configuration::GetInstance().GetDatabaseDir())
                    {
                        columnsJSON["file_path_string_data_file"] =
                            path + name_ + SEPARATOR + tableName + SEPARATOR + columnName + FRAGMENT_DATA_EXTENSION;
                        BOOST_LOG_TRIVIAL(debug)
                            << "Database: Default fragment file path ( "
                            << path + name_ + SEPARATOR + tableName + SEPARATOR + columnName + FRAGMENT_DATA_EXTENSION
                            << " ) has been persisted for column " << columnName << " of table "
                            << tableName << " of database " << name_ << ".";
                    }
                    else
                    {
                        columnsJSON["file_path_string_data_file"] = fileFragmentPath;
                        BOOST_LOG_TRIVIAL(debug)
                            << "Database: Specific fragment file path ( " << fileFragmentPath
                            << " ) has been persisted for column " << columnName << " of table "
                            << tableName << " of database " << name_ << ".";
                    }

                    if (encoding.size() == 0 || encoding == "undefined")
                    {
                        columnsJSON["encoding"] = "undefined";
                        BOOST_LOG_TRIVIAL(debug) << "Database: Default encoding (undefined) has "
                                                    "been persisted for column "
                                                 << columnName << " of table " << tableName
                                                 << " of database " << name_ << ".";
                    }
                    else
                    {
                        columnsJSON["encoding"] = encoding;
                        BOOST_LOG_TRIVIAL(debug)
                            << "Database: Specific encoding ( " << encoding
                            << " ) has been persisted for column " << columnName << " of table "
                            << tableName << " of database " << name_ << ".";
                    }
                }

                switch (columnType)
                {
                case COLUMN_POLYGON:
                {
                    const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon =
                        dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

                    columnsJSON["default_entry_value"] =
                        POLYGON_DEFAULT_VALUE; // We need to hardcode it due to Google Protobuffers
                }
                break;

                case COLUMN_POINT:
                {
                    const ColumnBase<ColmnarDB::Types::Point>& colPoint =
                        dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

                    columnsJSON["default_entry_value"] =
                        PointFactory::WktFromPoint(colPoint.GetDefaultValue());
                }
                break;

                case COLUMN_STRING:
                {
                    const ColumnBase<std::string>& colString =
                        dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

                    columnsJSON["default_entry_value"] = colString.GetDefaultValue();
                }
                break;

                case COLUMN_INT8_T:
                {
                    const ColumnBase<int8_t>& colBool =
                        dynamic_cast<const ColumnBase<int8_t>&>(*(column.second));

                    columnsJSON["default_entry_value"] = std::to_string(colBool.GetDefaultValue());
                }
                break;

                case COLUMN_INT:
                {
                    const ColumnBase<int32_t>& colInt =
                        dynamic_cast<const ColumnBase<int32_t>&>(*(column.second));

                    columnsJSON["default_entry_value"] = std::to_string(colInt.GetDefaultValue());
                }
                break;

                case COLUMN_LONG:
                {
                    const ColumnBase<int64_t>& colLong =
                        dynamic_cast<const ColumnBase<int64_t>&>(*(column.second));

                    columnsJSON["default_entry_value"] = std::to_string(colLong.GetDefaultValue());
                }
                break;

                case COLUMN_FLOAT:
                {
                    const ColumnBase<float>& colFloat =
                        dynamic_cast<const ColumnBase<float>&>(*(column.second));

                    columnsJSON["default_entry_value"] = std::to_string(colFloat.GetDefaultValue());
                }
                break;

                case COLUMN_DOUBLE:
                {
                    const ColumnBase<double>& colDouble =
                        dynamic_cast<const ColumnBase<double>&>(*(column.second));

                    columnsJSON["default_entry_value"] = std::to_string(colDouble.GetDefaultValue());
                }
                break;
                }

                columnsJSON["nullable"] = column.second->GetIsNullable();
                columnsJSON["unique"] = column.second->GetIsUnique();
                columnsJSON["hidden"] = false;
                columnsArrayJSON.append(columnsJSON);
            }

            tableJSON["columns"] = columnsArrayJSON;
            tableArrayJSON.append(tableJSON);
        }
        rootJSON["tables"] = tableArrayJSON;

        writer->write(rootJSON, &dbFile);
        dbFile.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "ERROR: Database: PersistOnlyDbFile() - Could not open file "
                                 << filePath << " for writing. Persisting "
                                 << " file was not successful. Check if the process "
                                    "have write access into the folder or file.";
    }
}

/// <summary>
/// Save database from memory to disk. Rewrites all database files.
/// </summary>
void Database::Persist()
{
    BOOST_LOG_TRIVIAL(info) << "Database: Saving database with name: " << name_ << " and "
                            << tables_.size() << " tables.";

    PersistOnlyDbFile();

    // write files COLUMN_DATA_EXTENSION:
    for (auto& table : tables_)
    {
        auto& columns = table.second.GetColumns();

        std::vector<std::thread> threads;

        for (const auto& column : columns)
        {
            threads.emplace_back(Database::WriteColumn, std::ref(column), name_, std::ref(table.second));
        }

        for (int j = 0; j < columns.size(); j++)
        {
            threads[j].join();
        }
    }

    if (boost::filesystem::exists(boost::filesystem::path(
            Configuration::GetInstance().GetDatabaseDir() + name_ + DB_EXTENSION)))
    {
        BOOST_LOG_TRIVIAL(info) << "Database: Database " << name_ << " was successfully saved into disk.";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info)
            << "Database: Database "
            << name_ << " was NOT saved into disk. Check if you have write access into the destination folder.";
    }
}

/// <summary>
/// Save modified blocks of data of all loaded database to disk.
/// </summary>
void Database::SaveModifiedToDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    BOOST_LOG_TRIVIAL(info)
        << "Database: Saving modified columns of loaded databases to disk has started...";
    auto path = Configuration::GetInstance().GetDatabaseDir().c_str();
    for (auto& database : Context::getInstance().GetLoadedDatabases())
    {
        auto& tablesHashMap = database.second->GetTables();

        for (auto& tablePair : tablesHashMap)
        {
            database.second->PersistOnlyModified(tablePair.first);
        }
    }
    BOOST_LOG_TRIVIAL(info)
        << "Database: Saving modified columns of loaded databases to disk has finished.";
}

/// <summary>
/// Save modified blocks of the specified table from memory to disk.
/// </summary>
/// <param name="tableName">Name of the table which modified blocks of data will be saved.</param>
void Database::PersistOnlyModified(const std::string tableName)
{
    // always persist at least db file
    PersistOnlyDbFile();

    // write files COLUMN_DATA_EXTENSION:
    auto& table = tables_.at(tableName);

    BOOST_LOG_TRIVIAL(info) << "Database: Saving only modified blocks of table named " << tableName
                            << " of database named " << name_ << ".";

    const auto& columns = table.GetColumns();

    std::vector<std::thread> threads;

    // fork threads for writing columns:
    for (auto& column : columns)
    {
        const int32_t type = column.second->GetColumnType();

        switch (type)
        {
        case COLUMN_POLYGON:
        {
            const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();
            std::string fileFragmentPath = column.second->GetFileFragmentPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileFragmentPath.size() == 0 || fileFragmentPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                   SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                   FRAGMENT_DATA_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);
            std::ifstream colFragDataFile(fileFragmentPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colPolygon.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint64_t strPolDataPos = UINT64_MAX; // this value is there just for debug purposes

                    uint32_t blockIndex = block->GetIndex();
                    colFragDataFile.seekg(0, colFragDataFile.end);

                    // we will persist new block at the end of FRAGMENT_DATA_EXTENSION file:
                    uint64_t blockPosition = colFragDataFile.tellg();

                    if (blockIndex != UINT32_MAX)
                    {
                        /* the block has been persisted at least once, so we need to mark all its current fragments as invalid
                           (we will persist the modified data as new fragments at the end of the file) */

                        uint64_t i = 0;
                        colAddressFile.seekg(0, colAddressFile.end);
                        uint64_t colAddressFileLength = colAddressFile.tellg();
                        colAddressFile.seekg(0, colAddressFile.beg);
                        while (i < colAddressFileLength)
                        {
                            uint32_t currentBlockPosition;
                            colAddressFile.read(reinterpret_cast<char*>(&currentBlockPosition), sizeof(uint32_t));

                            if (currentBlockPosition == blockIndex)
                            {
                                // mark fragment as invalid:
                                uint32_t value = UINT32_MAX;
                                colAddressFile.seekp(colAddressFile.tellg() -
                                                     static_cast<int64_t>(sizeof(uint32_t)));
                                colAddressFile.write(reinterpret_cast<char*>(&value), sizeof(uint32_t));
                                break;
                            }

                            i += sizeof(uint32_t);
                        }

                        i = 0;
                        colDataFile.seekg(0, colDataFile.end);
                        uint64_t colDataFileLength = colDataFile.tellg();
                        colDataFile.seekg(0, colDataFile.beg);
                        while (i < colDataFileLength)
                        {
                            uint32_t readBlockIndex;
                            colDataFile.read(reinterpret_cast<char*>(&readBlockIndex), sizeof(uint32_t));

                            if (readBlockIndex == blockIndex)
                            {
                                strPolDataPos = static_cast<uint64_t>(colDataFile.tellg()) - sizeof(uint32_t);
                                break;
                            }

                            int32_t readGroupId;
                            colDataFile.read(reinterpret_cast<char*>(&readGroupId), sizeof(int32_t));
                            if (column.second->GetIsNullable())
                            {
                                // TODO toto bude treba zmenit po zmene NullBit masiek z 8 bitov na 64 bitov
                                int32_t nullBitMaskLength;
                                std::unique_ptr<int8_t[]> nullBitMask = nullptr;
                                int32_t nullBitMaskAllocationSize =
                                    ((table.GetBlockSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));

                                colDataFile.read(reinterpret_cast<char*>(&nullBitMaskLength),
                                                 sizeof(int32_t)); // read nullBitMask length
                                nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                                colDataFile.read(reinterpret_cast<char*>(nullBitMask.get()),
                                                 nullBitMaskLength); // read nullBitMask
                            }
                            uint64_t readEntriesCount;
                            colDataFile.read(reinterpret_cast<char*>(&readEntriesCount), sizeof(uint64_t));
                        }
                    }

                    threads.emplace_back(WriteBlockPolygonType, std::ref(table), std::ref(column),
                                         std::ref(*block), blockPosition, strPolDataPos, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
            colFragDataFile.close();
        }
        break;

        case COLUMN_POINT:
        {
            const ColumnBase<ColmnarDB::Types::Point>& colPoint =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colPoint.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<ColmnarDB::Types::Point>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;

        case COLUMN_STRING:
        {
            const ColumnBase<std::string>& colStr =
                dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();
            std::string fileFragmentPath = colStr.GetFileFragmentPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileFragmentPath.size() == 0 || fileFragmentPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                   SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                   FRAGMENT_DATA_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);
            std::ifstream colFragDataFile(fileFragmentPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colStr.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint64_t strPolDataPos = UINT64_MAX; // this value is there just for debug purposes

                    uint32_t blockIndex = block->GetIndex();
                    colFragDataFile.seekg(0, colFragDataFile.end);

                    // we will persist new block at the end of FRAGMENT_DATA_EXTENSION file:
                    uint64_t blockPosition = colFragDataFile.tellg();

                    if (blockIndex != UINT32_MAX)
                    {
                        /* the block has been persisted at least once, so we need to mark all its current fragments as invalid
                           (we will persist the modified data as new fragments at the end of the file) */


                        uint64_t i = 0;
                        colAddressFile.seekg(0, colAddressFile.end);
                        uint64_t colAddressFileLength = colAddressFile.tellg();
                        colAddressFile.seekg(0, colAddressFile.beg);

                        while (i < colAddressFileLength)
                        {
                            uint32_t currentBlockPosition;
                            colAddressFile.read(reinterpret_cast<char*>(&currentBlockPosition), sizeof(uint32_t));

                            if (currentBlockPosition == blockIndex)
                            {
                                // mark fragment as invalid:
                                uint32_t value = UINT32_MAX;
                                colAddressFile.seekp(colAddressFile.tellg() -
                                                     static_cast<int64_t>(sizeof(uint32_t)));
                                colAddressFile.write(reinterpret_cast<char*>(&value), sizeof(uint32_t));
                                break;
                            }

                            i += sizeof(uint32_t);
                        }

                        i = 0;
                        colDataFile.seekg(0, colDataFile.end);
                        uint64_t colDataFileLength = colDataFile.tellg();
                        colDataFile.seekg(0, colDataFile.beg);
                        while (i < colDataFileLength)
                        {
                            uint32_t readBlockIndex;
                            colDataFile.read(reinterpret_cast<char*>(&readBlockIndex), sizeof(uint32_t));

                            if (readBlockIndex == blockIndex)
                            {
                                strPolDataPos = static_cast<uint64_t>(colDataFile.tellg()) - sizeof(uint32_t);
                                break;
                            }

                            int32_t readGroupId;
                            colDataFile.read(reinterpret_cast<char*>(&readGroupId), sizeof(int32_t));
                            if (column.second->GetIsNullable())
                            {
                                // TODO toto bude treba zmenit po zmene NullBit masiek z 8 bitov na 64 bitov
                                int32_t nullBitMaskLength;
                                std::unique_ptr<int8_t[]> nullBitMask = nullptr;
                                int32_t nullBitMaskAllocationSize =
                                    ((table.GetBlockSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));

                                colDataFile.read(reinterpret_cast<char*>(&nullBitMaskLength),
                                                 sizeof(int32_t)); // read nullBitMask length
                                nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                                colDataFile.read(reinterpret_cast<char*>(nullBitMask.get()),
                                                 nullBitMaskLength); // read nullBitMask
                            }
                            uint64_t readEntriesCount;
                            colDataFile.read(reinterpret_cast<char*>(&readEntriesCount), sizeof(uint64_t));
                        }
                    }

                    threads.emplace_back(WriteBlockStringType, std::ref(table), std::ref(column),
                                         std::ref(*block), blockPosition, strPolDataPos, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
            colFragDataFile.close();
        }
        break;

        case COLUMN_INT8_T:
        {
            const ColumnBase<int8_t>& colInt = dynamic_cast<const ColumnBase<int8_t>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colInt.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<int8_t>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;

        case COLUMN_INT:
        {
            const ColumnBase<int32_t>& colInt = dynamic_cast<const ColumnBase<int32_t>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colInt.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<int32_t>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;

        case COLUMN_LONG:
        {
            const ColumnBase<int64_t>& colLong = dynamic_cast<const ColumnBase<int64_t>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colLong.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<int64_t>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;

        case COLUMN_FLOAT:
        {
            const ColumnBase<float>& colFloat = dynamic_cast<const ColumnBase<float>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colFloat.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<float>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;

        case COLUMN_DOUBLE:
        {
            const ColumnBase<double>& colDouble = dynamic_cast<const ColumnBase<double>&>(*(column.second));

            std::string fileDataPath = column.second->GetFileDataPath();
            std::string fileAddressPath = column.second->GetFileAddressPath();

            // default data path if not specified by user:
            if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
            }

            // default data path if not specified by user:
            if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
            {
                fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + name_ +
                                  SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                  COLUMN_ADDRESS_EXTENSION;
            }

            // if the file does not exists, create it, because fstream need s file to exists before opening it:
            if (!boost::filesystem::exists(fileAddressPath))
            {
                std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
                colAddressFile.close();
            }

            std::fstream colAddressFile(fileAddressPath, std::ios::app | std::ios::binary);
            std::ifstream colDataFile(fileDataPath, std::ios::binary);

            // for each block of the column, check if it needs to be persisted and if so, persist it into disk:
            for (const auto& block : colDouble.GetBlocksList())
            {
                if (block->GetSaveNecessary())
                {
                    uint32_t blockIndex = block->GetIndex();
                    uint64_t blockPosition;

                    /* if this condition is true, that means, this block has never been persisted on
                       disk before, so we need to update also COLUMN_ADDRESS_EXTENSION file */
                    if (blockIndex == UINT32_MAX)
                    {
                        // add blockPosition (new value) at the end of an COLUMN_ADDRESS_EXTENSION file:
                        colAddressFile.seekp(0, colAddressFile.end); // seekp() - seek put position
                        colDataFile.seekg(0, colDataFile.end);
                        blockPosition = colDataFile.tellg();
                        colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }
                    else
                    {
                        // read the position of a block which has been persisted before and so the disk space is allocated already:
                        colAddressFile.seekg(blockIndex * sizeof(int64_t)); // seekg() - seek get position
                        colAddressFile.read(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));
                    }

                    threads.emplace_back(WriteBlockNumericTypes<double>, std::ref(table),
                                         std::ref(column), std::ref(*block), blockPosition, name_);
                }
            }

            colAddressFile.close();
            colDataFile.close();
        }
        break;
        }
    }

    // join threads:
    for (int j = 0; j < threads.size(); j++)
    {
        threads[j].join();
    }
}

/// <summary>
/// Save all databases currently loaded in memory to disk. Rewrites all loaded databases' files.
/// </summary>
void Database::SaveAllToDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    BOOST_LOG_TRIVIAL(info) << "Database: Saving all loaded databases to disk has started...";
    auto path = Configuration::GetInstance().GetDatabaseDir().c_str();
    for (auto& database : Context::getInstance().GetLoadedDatabases())
    {
        database.second->Persist();
    }
    BOOST_LOG_TRIVIAL(info) << "Database: Saving loaded databases to disk has finished.";
}

/// <summary>
/// Load databases from disk storage.
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
            if (extension == DB_EXTENSION)
            {
                auto database =
                    Database::LoadDatabase(p.path().filename().stem().generic_string().c_str(), path.c_str());

                if (database != nullptr)
                {
                    Context::getInstance().GetLoadedDatabases().insert({database->name_, database});
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Database: Directory " << path << " does not exists.";
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

    if (boost::filesystem::remove(path + name_ + DB_EXTENSION))
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

        PersistOnlyDbFile();
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning)
            << "Database: Renaming table: Main (" << DB_EXTENSION << ") file of db " << name_
            << " was NOT removed from disk. No such file (if the database was not yet saved, "
               "ignore this warning) or no write access.";
    }
}

/// <summary>
/// Delete database from disk. Deletes DB_EXTENSION and COLUMN_DATA_EXTENSION files which belong
/// to the specified database. Database is not deleted from memory.
/// </summary>
void Database::DeleteDatabaseFromDisk()
{
    std::unique_lock<std::mutex> lock(dbFilesMutex_);
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    // std::cout << "DeleteDatabaseFromDisk path: " << path << std::endl;
    if (boost::filesystem::exists(path))
    {
        // Delete main DB_EXTENSION file
        if (boost::filesystem::remove(path + name_ + DB_EXTENSION))
        {
            BOOST_LOG_TRIVIAL(info) << "Database: Main (" << DB_EXTENSION << ") file of db "
                                    << name_ << " was successfully removed from disk.";
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Database: Main (" << DB_EXTENSION << ") file of db "
                                       << name_ << " was NOT removed from disk. No such file or write access.";
        }

        // Delete tables and columns
        std::string prefix(path + name_ + SEPARATOR);
        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            // delete files which starts with prefix of db name:
            if (!p.path().string().compare(0, prefix.size(), prefix))
            {
                if (boost::filesystem::remove(p.path().string().c_str()))
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: File " << p.path().string()
                                            << " was successfully removed from disk.";
                }
                else
                {
                    BOOST_LOG_TRIVIAL(warning)
                        << "Database: File " << p.path().string()
                        << " was NOT removed from disk. No such file or write access.";
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Database: Directory " << path << " does not exists.";
    }
}

/// <summary>
/// <param name="tableName">Name of the table to be deleted.</param>
/// Delete table from disk. Deletes COLUMN_DATA_EXTENSION files which belong to the specified
/// table of currently loaded database. To alter DB_EXTENSION file, this action also calls a
/// function PersistOnlyDbFile(). Table needs to be deleted from memory before calling this
/// method, so that DB_EXTENSION file can be updated correctly.
/// </summary>
void Database::DeleteTableFromDisk(const char* tableName)
{
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    if (boost::filesystem::exists(path))
    {
        std::string prefix(path + name_ + SEPARATOR + std::string(tableName) + SEPARATOR);

        for (auto& p : boost::filesystem::directory_iterator(path))
        {
            // delete files which starts with prefix of db name and table name:
            if (!p.path().string().compare(0, prefix.size(), prefix))
            {
                if (boost::filesystem::remove(p.path().string().c_str()))
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: File " << p.path().string() << " from database "
                                            << name_ << " was successfully removed from disk.";
                }
                else
                {
                    BOOST_LOG_TRIVIAL(warning)
                        << "Database: File " << p.path().string()
                        << " was NOT removed from disk. No such file or write access.";
                }
            }
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Database: Directory " << path << " does not exists.";
    }

    // persist only db file, so that changes are saved, BUT PERSIST ONLY if there already is a DB_EXTENSION file, so it is not only in memory
    if (boost::filesystem::exists(path + name_ + DB_EXTENSION))
    {
        PersistOnlyDbFile();
    }
}

/// <summary>
/// <param name="tableName">Name of the table which have the specified column that will be
/// deleted.</param> <param name="columnName">Name of the column file without the
/// COLUMN_DATA_EXTENSION suffix that will be deleted.</param> Delete column of a table. Deletes
/// single COLUMN_DATA_EXTENSION file which belongs to specified column and specified table. To
/// alter DB_EXTENSION file, this action also calls a function Persist. Column needs to be deleted
/// from memory before calling this method, so that DB_EXTENSION file can be updated correctly.
/// </summary>
void Database::DeleteColumnFromDisk(const char* tableName, const char* columnName)
{
    auto& path = Configuration::GetInstance().GetDatabaseDir();

    std::string filePath = path + name_ + SEPARATOR + std::string(tableName) + SEPARATOR +
                           std::string(columnName) + COLUMN_DATA_EXTENSION;

    if (boost::filesystem::exists(filePath))
    {
        if (boost::filesystem::remove(filePath.c_str()))
        {
            BOOST_LOG_TRIVIAL(info) << "Database: Column " << columnName << " from table " << tableName
                                    << " from database " << name_ << " was successfully removed from disk.";
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: File " << filePath
                << " was NOT removed from disk. No such file or write access.";
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Database: File " << path << " does not exists.";
    }

    // persist only db file, so that changes are saved, BUT PERSIST ONLY if there already is a DB_EXTENSION file, so it is not only in memory
    if (boost::filesystem::exists(path + name_ + DB_EXTENSION))
    {
        PersistOnlyDbFile();
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
        BOOST_LOG_TRIVIAL(info)
            << "Database: The block size of the table named: " << tableName
            << " WILL BE changed from " << table.GetBlockSize() << " to " << newBlockSize << ".";

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

        // delete (original) COLUMN_DATA_EXTENSION files with old block size which are persisted on disk, if they are persisted
        DeleteTableFromDisk(tableName.c_str());

        // initialize sorting Columns to be empty
        newTableHashMap->second.SetSortingColumns(std::vector<std::string>());

        // delete original table from memory
        tables_.erase(tableName);

        RenameTable(("temp_" + tableName).c_str(), tableName);

        // save all changes to disk
        Persist();

        BOOST_LOG_TRIVIAL(info)
            << "Database: The block size of the table named: " << tableName
            << " HAS BEEN changed from " << table.GetBlockSize() << " to " << newBlockSize << ".";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info)
            << "Database: The new block size of the table named: " << tableName
            << " was the same as the current block size, so it has not been changed.";
    }
}

/// <summary>
/// Load database from disk into memory.
/// </summary>
/// <param name="fileDbName">Name of the database file without the DB_EXTENSION suffix.</param>
/// <param name="path">Path to directory in which database files are.</param>
/// <returns>Shared pointer of database.</returns>
std::shared_ptr<Database> Database::LoadDatabase(const char* fileDbName, const char* path)
{
    const std::string filePath = std::string(path) + std::string(fileDbName) + DB_EXTENSION;

    // read file DB_EXTENSION
    std::ifstream dbFile(filePath, std::ios::binary);

    dbFile.seekg(0, dbFile.end);
    const size_t fileSize = dbFile.tellg();
    if (fileSize != 0)
    {
        Json::Value root;
        Json::CharReaderBuilder builder;
        JSONCPP_STRING errs;

        dbFile.seekg(0, dbFile.beg);

        if (!parseFromStream(builder, dbFile, &root, &errs))
        {
            BOOST_LOG_TRIVIAL(error)
                << "Database: Cannot construct database from JSON file: " << filePath << ".";
            return nullptr;
        }

        BOOST_LOG_TRIVIAL(info) << "Database: Loading database from: " << filePath << ".";

        int32_t persistenceFormatVersion =
            root["persistence_format_version"].asInt(); // read persistence format version

        if (persistenceFormatVersion != Database::PERSISTENCE_FORMAT_VERSION)
        {
            BOOST_LOG_TRIVIAL(warning)
                << "WARNING: Database: Database persistence format version is different in "
                   "database file: "
                << filePath << ". The persisted database files are in persistence format version: " << persistenceFormatVersion
                << " the current persistence format version in this version of database core is: "
                << Database::PERSISTENCE_FORMAT_VERSION
                << ". There is going to be coversion to the database core format verion. "
                << "The database files on disk will be changed after successful persistence.";
        }

        const std::string dbName = root["database_name"].asString(); // read db name

        int32_t databaseBlockSize = root["database_default_block_size"].asInt(); // read block size

        Json::Value tablesArray(Json::arrayValue);
        tablesArray = root["tables"];

        std::shared_ptr<Database> database = std::make_shared<Database>(dbName.c_str(), databaseBlockSize);

        for (Json::Value tableJSON : tablesArray)
        {
            Json::Value indexColumnArray(Json::arrayValue);
            Json::Value columnArray(Json::arrayValue);

            const std::string tableName = tableJSON["table_name"].asString();
            const int32_t tableBlockSize = tableJSON["table_block_size"].asInt();
            const int32_t tableSaveInterval = tableJSON["save_interval_ms"].asInt();

            BOOST_LOG_TRIVIAL(info)
                << "Database: Block size for table: " + tableName +
                       " has been loaded and it's value is: " + std::to_string(tableBlockSize) + ".";

            database->tables_.emplace(
                std::make_pair(tableName, Table(database, tableName.c_str(), tableBlockSize)));

            indexColumnArray = tableJSON["index_columns"];
            columnArray = tableJSON["columns"];

            std::vector<std::string> columnNames;
            std::vector<std::string> sortingColumnNames;

            for (Json::Value indexColumnJSON : indexColumnArray)
            {
                const std::string sortingColumnName = indexColumnJSON["index_column_name"].asString();
                sortingColumnNames.push_back(sortingColumnName);
            }

            auto& table = database->tables_.at(tableName);
            table.SetSortingColumns(sortingColumnNames);
            table.SetSaveInterval(tableSaveInterval);

            std::vector<std::thread> threads;

            for (Json::Value columnJSON : columnArray)
            {
                const std::string columnName = columnJSON["column_name"].asString();
                const int32_t columnType = columnJSON["column_type"].asInt();
                const std::string filePathAddressFile = columnJSON["file_path_address_file"].asString();
                const std::string filePathDataFile = columnJSON["file_path_data_file"].asString();

                // usable just for COLUMN_STRING and COLUMN_POLYGON:
                std::string filePathStrDataFile = "";
                std::string encoding = "";
                if (columnType == COLUMN_STRING || columnType == COLUMN_POLYGON)
                {
                    filePathStrDataFile = columnJSON["file_path_string_data_file"].asString();
                    encoding = columnJSON["encoding"].asString();
                }

                const std::string defaultValue = columnJSON["default_entry_value"].asString();
                const bool isNullable = columnJSON["nullable"].asBool();
                const bool isUnique = columnJSON["unique"].asBool();
                const bool isHidden = columnJSON["hidden"].asBool();

                columnNames.push_back(columnName);
                threads.emplace_back(Database::LoadColumn, filePath, filePathAddressFile, filePathDataFile,
                                     filePathStrDataFile, encoding, persistenceFormatVersion, columnType,
                                     isNullable, isUnique, defaultValue, std::ref(table), columnName);
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
        BOOST_LOG_TRIVIAL(error) << "Database: File " + filePath + " is empty and so cannot be loaded.";
        return nullptr;
    }
}

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
/// <param name="table">Instance of table into which the column
/// should be added.</param><param name="columnName">Names of particular column.</param>
void Database::LoadColumn(const std::string fileDbPath,
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
                          const std::string columnName)
{
    std::ifstream colFile(fileDataPath, std::ios::binary);
    std::ifstream colAddressFile(fileAddressPath, std::ios::binary);

    colAddressFile.seekg(0, colAddressFile.end);
    const size_t fileAddressSize = colAddressFile.tellg();
    colAddressFile.seekg(0, colAddressFile.beg);

    colFile.seekg(0, colFile.end);
    const size_t fileDataSize = colFile.tellg();
    colFile.seekg(0, colFile.beg);

    BOOST_LOG_TRIVIAL(info) << "Database: Loading " << COLUMN_DATA_EXTENSION
                            << " file with name : " << fileDataPath << ".";

    int32_t nullBitMaskAllocationSize =
        ((table.GetBlockSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));

    switch (type)
    {
    case COLUMN_POLYGON:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_POLYGON, isNullable, isUnique);
        auto& columnPolygon =
            dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*table.GetColumns().at(columnName));
        std::vector<int32_t> fragBlockIndices; // position: fragment index, value: block index
        std::ifstream fragFile(fileFragmentPath, std::ios::binary);

        columnPolygon.SetFileAddressPath(fileAddressPath);
        columnPolygon.SetFileDataPath(fileDataPath);
        columnPolygon.SetFileFragmentPath(fileFragmentPath);
        columnPolygon.SetEncoding(encoding);
        columnPolygon.SetDefaultValue(ComplexPolygonFactory::FromWkt(defaultValue));

        if (fileAddressSize > 0)
        {
            while (!colAddressFile.eof())
            {
                uint32_t tempBlockIdx;
                colAddressFile.read(reinterpret_cast<char*>(&tempBlockIdx), sizeof(uint32_t)); // read fragment's block index

                // this is needed because of how EOF is checked:
                if (colAddressFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileAddressPath
                                            << " has finished successfully.";
                    break;
                }

                fragBlockIndices.push_back(tempBlockIdx);
            }
            colAddressFile.close();
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Address file " + fileAddressPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)


                auto& block = columnPolygon.AddBlock(groupId, false, index);
                std::vector<std::string> dataString;

                // for each fragment, read data, if it belongs to the current block index:
                for (int32_t fragIdx = 0; fragIdx < fragBlockIndices.size(); fragIdx++)
                {
                    if (index == fragBlockIndices[fragIdx])
                    {
                        fragFile.seekg(fragIdx * static_cast<int64_t>(FRAGMENT_SIZE_BYTES)); // seek the start of the fragment

                        int32_t readBytes = 0;

                        while (readBytes < FRAGMENT_SIZE_BYTES)
                        {
                            if (readBytes + sizeof(int32_t) <= FRAGMENT_SIZE_BYTES)
                            {

                                int32_t entryByteLength;
                                fragFile.read(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // read length of string entry data
                                readBytes += sizeof(int32_t);

                                // if entryByteLength > 0 that means, there is a valid entry, that is still in use and have to be read
                                if (entryByteLength > 0)
                                {
                                    std::unique_ptr<char[]> byteArray(new char[entryByteLength]);
                                    fragFile.read(byteArray.get(), entryByteLength);
                                    std::string entryDataString(byteArray.get());
                                    dataString.push_back(entryDataString);
                                    readBytes += entryByteLength * sizeof(char);
                                }
                                else
                                {
                                    // skip the invalid entry (just move pointer in file):
                                    entryByteLength = -entryByteLength;
                                    if (entryByteLength > 0) // to check, if it is not zero (it can be)
                                    {
                                        fragFile.seekg(static_cast<int64_t>(fragFile.tellg()) + entryByteLength);
                                        readBytes += entryByteLength * sizeof(char);
                                    }
                                }
                            }
                            else
                            {
                                // there is no left bytes for int32_t header in fragment:
                                readBytes = FRAGMENT_SIZE_BYTES;
                            }
                        }
                    }
                }

                std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;

                // convert string data into complex polygons:
                for (std::string dataStr : dataString)
                {
                    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(dataStr));
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + columnName + " has UNIQUE constraint and has not NOT NULL constraint");
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
                                "Loaded column: " + columnName + " has UNIQUE constraint and duplicate values: " +
                                ComplexPolygonFactory::WktFromPolygon(dataPolygon[i]));
                        }
                    }
                }

                block.InsertData(dataPolygon, false); // false, because we have this data persisted, we are reading them from disk

                block.SetNullBitmask(std::move(nullBitMask));
                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added ComplexPolygon block (" + fileDataPath + ") with data at index: "
                    << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }

        fragFile.close();
    }
    break;

    case COLUMN_POINT:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_POINT, isNullable, isUnique);

        auto& columnPoint =
            dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*table.GetColumns().at(columnName));

        columnPoint.SetFileAddressPath(fileAddressPath);
        columnPoint.SetFileDataPath(fileDataPath);
        columnPoint.SetDefaultValue(PointFactory::FromWkt(defaultValue));

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read number of entries
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed

                auto& block = columnPoint.AddBlock(groupId, false, index);
                block.SetIsCompressed(isCompressed);
                std::vector<ColmnarDB::Types::Point> dataPoint;

                if (dataLength > columnPoint.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
                                             ") from disk does not fit into existing block");
                    break;
                }

                // read actual entries:
                for (int32_t i = 0; i < dataLength; i++)
                {
                    float latitude;
                    float longitude;

                    colFile.read(reinterpret_cast<char*>(&latitude), sizeof(float)); // read latitude
                    colFile.read(reinterpret_cast<char*>(&longitude), sizeof(float)); // read longitude

                    ColmnarDB::Types::Point entryDataPoint = PointFactory::FromLatLon(latitude, longitude);
                    dataPoint.push_back(entryDataPoint);
                }

                // skip empty entries:
                colFile.seekg(static_cast<int64_t>(colFile.tellg()) +
                              ((static_cast<int64_t>(columnPoint.GetBlockSize()) - dataLength) * 2 *
                               sizeof(float)));

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
                    }

                    for (int32_t i = 0; i < dataPoint.size(); i++)
                    {
                        if (!columnPoint.IsDuplicate(dataPoint[i]))
                        {
                            columnPoint.InsertIntoHashmap(dataPoint[i]);
                        }
                        else
                        {
                            throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and duplicate values: " +
                                                     PointFactory::WktFromPoint(dataPoint[i]));
                        }
                    }
                }
                block.InsertData(dataPoint, false);

                block.SetNullBitmask(std::move(nullBitMask));
                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Point block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    case COLUMN_STRING:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_STRING, isNullable, isUnique);
        auto& columnString = dynamic_cast<ColumnBase<std::string>&>(*table.GetColumns().at(columnName));
        std::vector<int32_t> fragBlockIndices; // position: fragment index, value: block index
        std::ifstream fragFile(fileFragmentPath, std::ios::binary);

        columnString.SetFileAddressPath(fileAddressPath);
        columnString.SetFileDataPath(fileDataPath);
        columnString.SetFileFragmentPath(fileFragmentPath);
        columnString.SetEncoding(encoding);
        columnString.SetDefaultValue(defaultValue);

        if (fileAddressSize > 0)
        {
            while (!colAddressFile.eof())
            {
                uint32_t tempBlockIdx;
                colAddressFile.read(reinterpret_cast<char*>(&tempBlockIdx), sizeof(uint32_t)); // read fragment's block index

                // this is needed because of how EOF is checked:
                if (colAddressFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileAddressPath
                                            << " has finished successfully.";
                    break;
                }

                fragBlockIndices.push_back(tempBlockIdx);
            }
            colAddressFile.close();
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Address file " + fileAddressPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)


                auto& block = columnString.AddBlock(groupId, false, index);
                std::vector<std::string> dataString;

                // for each fragment, read data, if it belongs to the current block index:
                for (int32_t fragIdx = 0; fragIdx < fragBlockIndices.size(); fragIdx++)
                {
                    if (index == fragBlockIndices[fragIdx])
                    {
                        fragFile.seekg(fragIdx * static_cast<int64_t>(FRAGMENT_SIZE_BYTES)); // seek the start of the fragment

                        int32_t readBytes = 0;

                        while (readBytes < FRAGMENT_SIZE_BYTES)
                        {
                            if (readBytes + sizeof(int32_t) <= FRAGMENT_SIZE_BYTES)
                            {

                                int32_t entryByteLength;
                                fragFile.read(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // read length of string entry data
                                readBytes += sizeof(int32_t);

                                // if entryByteLength > 0 that means, there is a valid entry, that is still in use and have to be read
                                if (entryByteLength > 0)
                                {
                                    std::unique_ptr<char[]> byteArray(new char[entryByteLength]);
                                    fragFile.read(byteArray.get(), entryByteLength);
                                    std::string entryDataString(byteArray.get());
                                    dataString.push_back(entryDataString);
                                    readBytes += entryByteLength * sizeof(char);
                                }
                                else
                                {
                                    // skip the invalid entry (just move pointer in file):
                                    entryByteLength = -entryByteLength;
                                    if (entryByteLength > 0) // to check, if it is not zero (it can be)
                                    {
                                        fragFile.seekg(static_cast<int64_t>(fragFile.tellg()) + entryByteLength);
                                        readBytes += entryByteLength * sizeof(char);
                                    }
                                }
                            }
                            else
                            {
                                // there is no left bytes for int32_t header in fragment:
                                readBytes = FRAGMENT_SIZE_BYTES;
                            }
                        }
                    }
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
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
                                "Loaded column: " + fileDataPath +
                                " has UNIQUE constraint and duplicate values: " + dataString[i]);
                        }
                    }
                }
                block.InsertData(dataString, false);

                block.SetNullBitmask(std::move(nullBitMask));
                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added String block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }

        fragFile.close();
    }
    break;

    case COLUMN_INT8_T:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_INT8_T, isNullable, isUnique);

        auto& columnInt = dynamic_cast<ColumnBase<int8_t>&>(*table.GetColumns().at(columnName));

        columnInt.SetFileAddressPath(fileAddressPath);
        columnInt.SetFileDataPath(fileDataPath);

        const std::string lowerStr = boost::algorithm::to_lower_copy(defaultValue);

        // just check if user did not write the bool default value as string 'true' or 'false':
        if (lowerStr == "true")
        {
            columnInt.SetDefaultValue(1);
        }
        else
        {
            if (lowerStr == "false")
            {
                columnInt.SetDefaultValue(0);
            }
            else
            {
                columnInt.SetDefaultValue(static_cast<int8_t>(std::stoi(defaultValue)));
            }
        }

        if (fileDataSize > 0)
        {

            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed
                int8_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int8_t)); // read statistics min
                int8_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int8_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int8_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int8_t)); // read statistics sum


                std::unique_ptr<int8_t[]> data =
                    std::unique_ptr<int8_t[]>(new int8_t[columnInt.GetBlockSize()]);
                std::unique_ptr<int8_t[]> emptyData =
                    std::unique_ptr<int8_t[]>(new int8_t[columnInt.GetBlockSize() - dataLength]);

                colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int8_t)); // read entry data
                colFile.read(reinterpret_cast<char*>(emptyData.get()),
                             (columnInt.GetBlockSize() - dataLength) * sizeof(int8_t)); // read empty entries as well

                if (dataLength > columnInt.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
                                             ") from disk does not fit into existing block");
                    break;
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
                    }
                    std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                  [&columnInt, &fileDataPath](int8_t& value) {
                                      if (!columnInt.IsDuplicate(value))
                                      {
                                          columnInt.InsertIntoHashmap(value);
                                      }
                                      else
                                      {
                                          throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and duplicate values: " +
                                                                   std::to_string(value));
                                      }
                                  });
                }

                auto& block = columnInt.AddBlock(std::move(data), dataLength, columnInt.GetBlockSize(),
                                                 groupId, false, isCompressed, false, false, index);
                block.SetNullBitmask(std::move(nullBitMask));
                block.setBlockStatistics(min, max, avg, sum, dataLength);

                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Int8 block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    case COLUMN_INT:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_INT, isNullable, isUnique);

        auto& columnInt = dynamic_cast<ColumnBase<int32_t>&>(*table.GetColumns().at(columnName));

        columnInt.SetFileAddressPath(fileAddressPath);
        columnInt.SetFileDataPath(fileDataPath);
        columnInt.SetDefaultValue(std::stoi(defaultValue));

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed
                int32_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int32_t)); // read statistics min
                int32_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int32_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int32_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int32_t)); // read statistics sum

                std::unique_ptr<int32_t[]> data =
                    std::unique_ptr<int32_t[]>(new int32_t[columnInt.GetBlockSize()]);
                std::unique_ptr<int32_t[]> emptyData =
                    std::unique_ptr<int32_t[]>(new int32_t[columnInt.GetBlockSize() - dataLength]);

                colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int32_t)); // read entry data
                colFile.read(reinterpret_cast<char*>(emptyData.get()),
                             (columnInt.GetBlockSize() - dataLength) * sizeof(int32_t)); // read empty entries as well

                if (dataLength > columnInt.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
                                             ") from disk does not fit into existing block");
                    break;
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
                    }
                    std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                  [&columnInt, &fileDataPath](int32_t& value) {
                                      if (!columnInt.IsDuplicate(value))
                                      {
                                          columnInt.InsertIntoHashmap(value);
                                      }
                                      else
                                      {
                                          throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and duplicate values: " +
                                                                   std::to_string(value));
                                      }
                                  });
                }

                auto& block = columnInt.AddBlock(std::move(data), dataLength, columnInt.GetBlockSize(),
                                                 groupId, false, isCompressed, false, false, index);
                block.SetNullBitmask(std::move(nullBitMask));
                block.setBlockStatistics(min, max, avg, sum, dataLength);

                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Int32 block (" + fileDataPath + ") with data at index : " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    case COLUMN_LONG:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_LONG, isNullable, isUnique);

        auto& columnLong = dynamic_cast<ColumnBase<int64_t>&>(*table.GetColumns().at(columnName));

        columnLong.SetFileAddressPath(fileAddressPath);
        columnLong.SetFileDataPath(fileDataPath);
        columnLong.SetDefaultValue(std::stol(defaultValue));

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed
                int64_t min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(int64_t)); // read statistics min
                int64_t max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(int64_t)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                int64_t sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(int64_t)); // read statistics sum

                std::unique_ptr<int64_t[]> data =
                    std::unique_ptr<int64_t[]>(new int64_t[columnLong.GetBlockSize()]);
                std::unique_ptr<int64_t[]> emptyData =
                    std::unique_ptr<int64_t[]>(new int64_t[columnLong.GetBlockSize() - dataLength]);

                colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(int64_t)); // read entry data
                colFile.read(reinterpret_cast<char*>(emptyData.get()),
                             (columnLong.GetBlockSize() - dataLength) * sizeof(int64_t)); // read empty entries as well

                if (dataLength > columnLong.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
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

                auto& block = columnLong.AddBlock(std::move(data), dataLength, columnLong.GetBlockSize(),
                                                  groupId, false, isCompressed, false, false, index);
                block.SetNullBitmask(std::move(nullBitMask));
                block.setBlockStatistics(min, max, avg, sum, dataLength);

                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Int64 block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    case COLUMN_FLOAT:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_FLOAT, isNullable, isUnique);

        auto& columnFloat = dynamic_cast<ColumnBase<float>&>(*table.GetColumns().at(columnName));

        columnFloat.SetFileAddressPath(fileAddressPath);
        columnFloat.SetFileDataPath(fileDataPath);
        columnFloat.SetDefaultValue(std::stof(defaultValue));

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed
                float min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(float)); // read statistics min
                float max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(float)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                float sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(float)); // read statistics sum

                std::unique_ptr<float[]> data =
                    std::unique_ptr<float[]>(new float[columnFloat.GetBlockSize()]);
                std::unique_ptr<float[]> emptyData =
                    std::unique_ptr<float[]>(new float[columnFloat.GetBlockSize() - dataLength]);

                colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(float)); // read entry data
                colFile.read(reinterpret_cast<char*>(emptyData.get()),
                             (columnFloat.GetBlockSize() - dataLength) * sizeof(float)); // read empty entries as well

                if (dataLength > columnFloat.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
                                             ") from disk does not fit into existing block");
                    break;
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
                    }
                    std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                  [&columnFloat, &fileDataPath](float& value) {
                                      if (!columnFloat.IsDuplicate(value))
                                      {
                                          columnFloat.InsertIntoHashmap(value);
                                      }
                                      else
                                      {
                                          throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and duplicate values: " +
                                                                   std::to_string(value));
                                      }
                                  });
                }

                auto& block = columnFloat.AddBlock(std::move(data), dataLength, columnFloat.GetBlockSize(),
                                                   groupId, false, isCompressed, false, false, index);
                block.SetNullBitmask(std::move(nullBitMask));
                block.setBlockStatistics(min, max, avg, sum, dataLength);

                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Float block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    case COLUMN_DOUBLE:
    {
        table.CreateColumn(columnName.c_str(), COLUMN_DOUBLE, isNullable, isUnique);

        auto& columnDouble = dynamic_cast<ColumnBase<double>&>(*table.GetColumns().at(columnName));

        columnDouble.SetFileAddressPath(fileAddressPath);
        columnDouble.SetFileDataPath(fileDataPath);
        columnDouble.SetDefaultValue(std::stod(defaultValue));

        if (fileDataSize > 0)
        {
            while (!colFile.eof())
            {
                uint32_t index;
                colFile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // read block index

                int32_t groupId;
                colFile.read(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // read block groupId

                int32_t nullBitMaskLength;

                if (isNullable)
                {
                    colFile.read(reinterpret_cast<char*>(&nullBitMaskLength), sizeof(int32_t)); // read nullBitMask length
                }

                std::unique_ptr<int8_t[]> nullBitMask = nullptr;

                if (isNullable)
                {
                    nullBitMask = std::unique_ptr<int8_t[]>(new int8_t[nullBitMaskAllocationSize]);
                    colFile.read(reinterpret_cast<char*>(nullBitMask.get()), nullBitMaskLength); // read nullBitMask
                }

                // this is needed because of how EOF is checked:
                if (colFile.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Database: Loading of the file: " << fileDataPath
                                            << " has finished successfully.";
                    break;
                }

                uint64_t dataLength;
                colFile.read(reinterpret_cast<char*>(&dataLength),
                             sizeof(uint64_t)); // read data length (number of entries)
                bool isCompressed;
                colFile.read(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // read whether compressed
                double min;
                colFile.read(reinterpret_cast<char*>(&min), sizeof(double)); // read statistics min
                double max;
                colFile.read(reinterpret_cast<char*>(&max), sizeof(double)); // read statistics max
                float avg;
                colFile.read(reinterpret_cast<char*>(&avg), sizeof(float)); // read statistics avg
                double sum;
                colFile.read(reinterpret_cast<char*>(&sum), sizeof(double)); // read statistics sum

                std::unique_ptr<double[]> data =
                    std::unique_ptr<double[]>(new double[columnDouble.GetBlockSize()]);
                std::unique_ptr<double[]> emptyData =
                    std::unique_ptr<double[]>(new double[columnDouble.GetBlockSize() - dataLength]);

                colFile.read(reinterpret_cast<char*>(data.get()), dataLength * sizeof(double)); // read entry data
                colFile.read(reinterpret_cast<char*>(emptyData.get()),
                             (columnDouble.GetBlockSize() - dataLength) * sizeof(double)); // read empty entries as well

                if (dataLength > columnDouble.GetBlockSize())
                {
                    throw std::runtime_error("Loaded data (" + fileDataPath +
                                             ") from disk does not fit into existing block");
                    break;
                }

                if (isUnique)
                {
                    if (isNullable)
                    {
                        throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and has not NOT NULL constraint");
                    }
                    std::for_each(std::next(data.get(), 0), std::next(data.get(), dataLength),
                                  [&columnDouble, &fileDataPath](double& value) {
                                      if (!columnDouble.IsDuplicate(value))
                                      {
                                          columnDouble.InsertIntoHashmap(value);
                                      }
                                      else
                                      {
                                          throw std::runtime_error("Loaded column: " + fileDataPath + " has UNIQUE constraint and duplicate values: " +
                                                                   std::to_string(value));
                                      }
                                  });
                }

                auto& block = columnDouble.AddBlock(std::move(data), dataLength, columnDouble.GetBlockSize(),
                                                    groupId, false, isCompressed, false, false, index);
                block.SetNullBitmask(std::move(nullBitMask));
                block.setBlockStatistics(min, max, avg, sum, dataLength);

                BOOST_LOG_TRIVIAL(debug)
                    << "Database: Added Double block (" + fileDataPath + ") with data at index: " << index;
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Database: Data file " + fileDataPath + " is empty and so the loading will be skipped. "
                << "If the column should have zero blocks of data, this behavior is correct.";
        }
    }
    break;

    default:
        BOOST_LOG_TRIVIAL(error) << "Database: Unsupported data type (when loading database - "
                                 << fileDbPath << ") encountered type number: " << type;
        throw std::domain_error("Unsupported data type (when loading database - " + fileDbPath +
                                ") encountered type number: " + std::to_string(type));
    }

    colFile.close();
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
/// <param name="dbName">Name of the database.</param>
/// <param name="table">Name of the particular table.</param>
void Database::WriteColumn(const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                           const std::string dbName,
                           const Table& table)
{
    int32_t blockSize = table.GetBlockSize();
    std::string fileDataPath = column.second->GetFileDataPath();
    std::string fileAddressPath = column.second->GetFileAddressPath();
    const std::string tableName = table.GetName();

    // default data path if not specified by user:
    if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
    {
        fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName + SEPARATOR +
                       tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
    }

    // default data path if not specified by user:
    if (fileAddressPath.size() == 0 || fileAddressPath == Configuration::GetInstance().GetDatabaseDir())
    {
        fileAddressPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName + SEPARATOR +
                          tableName + SEPARATOR + column.second->GetName() + COLUMN_ADDRESS_EXTENSION;
    }

    std::ofstream colAddressFile(fileAddressPath, std::ios::binary);
    BOOST_LOG_TRIVIAL(debug) << "Database: Saving " << COLUMN_ADDRESS_EXTENSION
                             << " file with name : " << fileAddressPath << ".";

    std::ofstream colDataFile(fileDataPath, std::ios::binary);
    BOOST_LOG_TRIVIAL(debug) << "Database: Saving " << COLUMN_DATA_EXTENSION
                             << " file with name : " << fileDataPath << ".";

    if (colDataFile.is_open())
    {
        if (colAddressFile.is_open())
        {
            const int32_t type = column.second->GetColumnType();
            const bool isNullable = column.second->GetIsNullable();
            const bool isUnique = column.second->GetIsUnique();

            switch (type)
            {
            case COLUMN_POLYGON:
            {
                uint32_t index = 0;

                const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon =
                    dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

                std::string fileFragmentPath = colPolygon.GetFileFragmentPath();

                // default data path if not specified by user:
                if (fileFragmentPath.size() == 0 ||
                    fileFragmentPath == Configuration::GetInstance().GetDatabaseDir())
                {
                    fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() +
                                       dbName + SEPARATOR + tableName + SEPARATOR +
                                       column.second->GetName() + FRAGMENT_DATA_EXTENSION;
                }

                std::ofstream colFragDataFile(fileFragmentPath, std::ios::binary);

                for (const auto& block : colPolygon.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug)
                        << "Database: Saving block of ComplexPolygon data with index = " << index;

                    auto data = block->GetData();
                    int32_t groupId = block->GetGroupId();
                    size_t blockCurrentSize = block->GetSize();
                    int64_t dataByteSize = 0;

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write block index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write group id (binary index)

                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (block->GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }

                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize), sizeof(uint64_t)); // write number of entries

                    if (colFragDataFile.is_open())
                    {
                        bool newFragment = true;

                        // write string data (entries in WKT format) into polygon fragment data file:
                        for (int32_t i = 0; i < blockCurrentSize; i++)
                        {
                            // write block index (ID) into COLUMN_ADDRESS_EXTENSION file for a new fragment
                            if (newFragment)
                            {
                                colAddressFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t));

                                newFragment = false;
                            }

                            // transform protobuf message into WKT strings:
                            std::string wktPolygon = ComplexPolygonFactory::WktFromPolygon(data[i]);

                            // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                            dataByteSize += wktPolygon.length() + 1 + sizeof(int32_t);

                            if (dataByteSize <= FRAGMENT_SIZE_BYTES)
                            {
                                // writing entries that fit into a fragment
                                int32_t entryByteLength = wktPolygon.length() + 1; // +1 because '\0'

                                colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(wktPolygon.c_str(), entryByteLength); // write entry data
                            }
                            else
                            {
                                // there is still some data which will be saved into next fragment:
                                // padding the not full fragment to it's maximum size, so the size of fragment is always fixed:
                                if (dataByteSize - (wktPolygon.length() + 1 + sizeof(int32_t)) < FRAGMENT_SIZE_BYTES)
                                {
                                    const int32_t freeSpaceByteLength =
                                        FRAGMENT_SIZE_BYTES -
                                        (dataByteSize - (wktPolygon.length() + 1 + sizeof(int32_t)));

                                    // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                                    if (freeSpaceByteLength >= sizeof(int32_t))
                                    {
                                        int32_t freeSpaceByteLengthNeg =
                                            -(freeSpaceByteLength - sizeof(int32_t));
                                        std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                        colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                              sizeof(int32_t)); // write entry length
                                        colFragDataFile.write(emptyStringData.c_str(),
                                                              freeSpaceByteLength - sizeof(int32_t)); // write empty data
                                    }
                                    else
                                    {
                                        // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                        // do not write int32_t header, because there is no space for it
                                        std::string emptyStringData(freeSpaceByteLength, '*');
                                        colFragDataFile.write(emptyStringData.c_str(),
                                                              freeSpaceByteLength); // write empty data
                                    }
                                }

                                // write the actual entry into another fragment (create a new fragment) and change data byte size:
                                // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                                dataByteSize = wktPolygon.length() + 1 + sizeof(int32_t);
                                newFragment = true;

                                // writing entries that fit into a fragment
                                int32_t entryByteLength = wktPolygon.length() + 1; // +1 because '\0'
                                colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(wktPolygon.c_str(), entryByteLength); // write entry data
                            }

                            // padding the not full fragment, when there is no data to be saved in another fragmet
                            if (i == blockCurrentSize - 1)
                            {
                                const int32_t freeSpaceByteLength = FRAGMENT_SIZE_BYTES - dataByteSize;

                                // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                                if (freeSpaceByteLength >= sizeof(int32_t))
                                {
                                    int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                                    std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                    colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                          sizeof(int32_t)); // write entry length
                                    colFragDataFile.write(emptyStringData.c_str(),
                                                          freeSpaceByteLength - sizeof(int32_t)); // write empty data
                                }
                                else
                                {
                                    // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                    // do not write int32_t header, because there is no space for it
                                    std::string emptyStringData(freeSpaceByteLength, '*');
                                    colFragDataFile.write(emptyStringData.c_str(),
                                                          freeSpaceByteLength); // write empty data
                                }
                            }
                        }
                    }
                    else
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: WriteColumn, case1 - Could not open file " +
                                   std::string(Configuration::GetInstance().GetDatabaseDir() +
                                               dbName + SEPARATOR + tableName + SEPARATOR +
                                               column.second->GetName() + FRAGMENT_DATA_EXTENSION) +
                                   " for writing. Persisting "
                            << FRAGMENT_DATA_EXTENSION
                            << " file was not successful. Check if the process "
                               "have write access into the folder or file.";
                    }

                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }

                colFragDataFile.close();
            }
            break;

            case COLUMN_POINT:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<ColmnarDB::Types::Point>& colPoint =
                    dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

                for (const auto& block : colPoint.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Point data with index = " << index;

                    auto data = block->GetData();
                    int32_t groupId = block->GetGroupId();
                    size_t blockCurrentSize = block->GetSize();
                    bool isCompressed = block->IsCompressed();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId

                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (block->GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }

                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed

                    // write entries:
                    for (size_t i = 0; i < blockCurrentSize; i++)
                    {
                        float latitude = data[i].geopoint().latitude();
                        float longitude = data[i].geopoint().longitude();

                        colDataFile.write(reinterpret_cast<char*>(&latitude), sizeof(float)); // write latitude
                        colDataFile.write(reinterpret_cast<char*>(&longitude), sizeof(float)); // write longitude
                    }

                    // padding to block size with std::numeric_limits<float>::max() values:
                    for (size_t i = blockCurrentSize; i < blockSize; i++)
                    {
                        float value = std::numeric_limits<float>::max();

                        colDataFile.write(reinterpret_cast<char*>(&value), sizeof(float)); // write latitude
                        colDataFile.write(reinterpret_cast<char*>(&value), sizeof(float)); // write longitude
                    }

                    const int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 2 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) +
                                     2 * blockSize * sizeof(float);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            case COLUMN_STRING:
            {
                uint32_t index = 0;

                const ColumnBase<std::string>& colStr =
                    dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

                std::string fileFragmentPath = colStr.GetFileFragmentPath();

                // default data path if not specified by user:
                if (fileFragmentPath.size() == 0 ||
                    fileFragmentPath == Configuration::GetInstance().GetDatabaseDir())
                {
                    fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() +
                                       dbName + SEPARATOR + tableName + SEPARATOR +
                                       column.second->GetName() + FRAGMENT_DATA_EXTENSION;
                }

                std::ofstream colFragDataFile(fileFragmentPath, std::ios::binary);

                for (const auto& block : colStr.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug)
                        << "Database: Saving block of String data with index = " << index;

                    auto data = block->GetData();
                    int32_t groupId = block->GetGroupId();
                    size_t blockCurrentSize = block->GetSize();
                    int64_t dataByteSize = 0;

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId

                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (block->GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }

                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)


                    if (colFragDataFile.is_open())
                    {

                        bool newFragment = true;

                        // write string data (entries) into string data file:
                        for (int32_t i = 0; i < blockCurrentSize; i++)
                        {
                            // write block index (ID) into COLUMN_ADDRESS_EXTENSION file for a new fragment
                            if (newFragment)
                            {
                                colAddressFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t));

                                newFragment = false;
                            }

                            // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                            dataByteSize += data[i].length() + 1 + sizeof(int32_t);

                            if (dataByteSize <= FRAGMENT_SIZE_BYTES)
                            {
                                // writing entries that fit into a fragment
                                int32_t entryByteLength = data[i].length() + 1; // +1 because '\0'

                                colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(data[i].c_str(), entryByteLength); // write entry data
                            }
                            else
                            {
                                // there is still some data which will be saved into next fragment:
                                // padding the not full fragment to it's maximum size, so the size of fragment is always fixed:
                                if (dataByteSize - (data[i].length() + 1 + sizeof(int32_t)) < FRAGMENT_SIZE_BYTES)
                                {
                                    const int32_t freeSpaceByteLength =
                                        FRAGMENT_SIZE_BYTES -
                                        (dataByteSize - (data[i].length() + 1 + sizeof(int32_t)));

                                    // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                                    if (freeSpaceByteLength >= sizeof(int32_t))
                                    {
                                        int32_t freeSpaceByteLengthNeg =
                                            -(freeSpaceByteLength - sizeof(int32_t));
                                        std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                        colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                              sizeof(int32_t)); // write entry length
                                        colFragDataFile.write(emptyStringData.c_str(),
                                                              freeSpaceByteLength - sizeof(int32_t)); // write empty data
                                    }
                                    else
                                    {
                                        // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                        // do not write int32_t header, because there is no space for it
                                        std::string emptyStringData(freeSpaceByteLength, '*');
                                        colFragDataFile.write(emptyStringData.c_str(),
                                                              freeSpaceByteLength); // write empty data
                                    }
                                }

                                // write the actual entry into another fragment (create a new fragment) and change data byte size:
                                // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                                dataByteSize = data[i].length() + 1 + sizeof(int32_t);
                                newFragment = true;

                                // writing entries that fit into a fragment
                                int32_t entryByteLength = data[i].length() + 1; // +1 because '\0'
                                colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(data[i].c_str(), entryByteLength); // write entry data
                            }

                            // padding the not full fragment, when there is no data to be saved in another fragmet
                            if (i == blockCurrentSize - 1)
                            {
                                const int32_t freeSpaceByteLength = FRAGMENT_SIZE_BYTES - dataByteSize;

                                // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                                if (freeSpaceByteLength >= sizeof(int32_t))
                                {
                                    int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                                    std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                    colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                          sizeof(int32_t)); // write entry length
                                    colFragDataFile.write(emptyStringData.c_str(),
                                                          freeSpaceByteLength - sizeof(int32_t)); // write empty data
                                }
                                else
                                {
                                    // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                    // do not write int32_t header, because there is no space for it
                                    std::string emptyStringData(freeSpaceByteLength, '*');
                                    colFragDataFile.write(emptyStringData.c_str(),
                                                          freeSpaceByteLength); // write empty data
                                }
                            }
                        }
                    }
                    else
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: WriteColumn, case2 - Could not open file " +
                                   std::string(Configuration::GetInstance().GetDatabaseDir() +
                                               dbName + SEPARATOR + tableName + SEPARATOR +
                                               column.second->GetName() + FRAGMENT_DATA_EXTENSION) +
                                   " for writing. Persisting "
                            << FRAGMENT_DATA_EXTENSION
                            << " file was not successful. Check if the process "
                               "have write access into the folder or file.";
                    }

                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }

                colFragDataFile.close();
            }
            break;

            case COLUMN_INT8_T:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<int8_t>& colInt = dynamic_cast<const ColumnBase<int8_t>&>(*(column.second));

                for (const auto& block : colInt.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Int8 data with index = " << index;

                    auto data = block->GetData();
                    size_t blockCurrentSize = block->GetSize();
                    std::unique_ptr<int8_t[]> emptyData(new int8_t[blockSize - blockCurrentSize]);
                    std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                              std::numeric_limits<int8_t>::max());
                    bool isCompressed = block->IsCompressed();
                    int32_t groupId = block->GetGroupId();
                    int8_t min = block->GetMin();
                    int8_t max = block->GetMax();
                    float avg = block->GetAvg();
                    int8_t sum = block->GetSum();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }
                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
                    colDataFile.write(reinterpret_cast<char*>(&min), sizeof(int8_t)); // write statistics min
                    colDataFile.write(reinterpret_cast<char*>(&max), sizeof(int8_t)); // write statistics max
                    colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(int8_t)); // write statistics sum
                    colDataFile.write(reinterpret_cast<const char*>(data),
                                      blockCurrentSize * sizeof(int8_t)); // write block of data
                    colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                                      (blockSize - blockCurrentSize) * sizeof(int8_t)); // write empty entries as well

                    int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 2 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) +
                                     sizeof(float) + 3 * sizeof(int8_t) + blockSize * sizeof(int8_t);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            case COLUMN_INT:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<int32_t>& colInt =
                    dynamic_cast<const ColumnBase<int32_t>&>(*(column.second));

                for (const auto& block : colInt.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Int32 data with index = " << index;

                    auto data = block->GetData();
                    size_t blockCurrentSize = block->GetSize();
                    std::unique_ptr<int32_t[]> emptyData(new int32_t[blockSize - blockCurrentSize]);
                    std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                              std::numeric_limits<int32_t>::max());
                    bool isCompressed = block->IsCompressed();
                    int32_t groupId = block->GetGroupId();
                    int32_t min = block->GetMin();
                    int32_t max = block->GetMax();
                    float avg = block->GetAvg();
                    int32_t sum = block->GetSum();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }
                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
                    colDataFile.write(reinterpret_cast<char*>(&min), sizeof(int32_t)); // write statistics min
                    colDataFile.write(reinterpret_cast<char*>(&max), sizeof(int32_t)); // write statistics max
                    colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(int32_t)); // write statistics sum
                    colDataFile.write(reinterpret_cast<const char*>(data),
                                      blockCurrentSize * sizeof(int32_t)); // write block of data
                    colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                                      (blockSize - blockCurrentSize) * sizeof(int32_t)); // write empty entries as well

                    int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 5 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) +
                                     sizeof(float) + blockSize * sizeof(int32_t);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            case COLUMN_LONG:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<int64_t>& colLong =
                    dynamic_cast<const ColumnBase<int64_t>&>(*(column.second));

                for (const auto& block : colLong.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Int64 data with index = " << index;

                    auto data = block->GetData();
                    size_t blockCurrentSize = block->GetSize();
                    std::unique_ptr<int64_t[]> emptyData(new int64_t[blockSize - blockCurrentSize]);
                    std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                              std::numeric_limits<int64_t>::max());
                    bool isCompressed = block->IsCompressed();
                    int32_t groupId = block->GetGroupId();
                    int64_t min = block->GetMin();
                    int64_t max = block->GetMax();
                    float avg = block->GetAvg();
                    int64_t sum = block->GetSum();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }
                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
                    colDataFile.write(reinterpret_cast<char*>(&min), sizeof(int64_t)); // write statistics min
                    colDataFile.write(reinterpret_cast<char*>(&max), sizeof(int64_t)); // write statistics max
                    colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(int64_t)); // write statistics sum
                    colDataFile.write(reinterpret_cast<const char*>(data),
                                      blockCurrentSize * sizeof(int64_t)); // write block of data
                    colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                                      (blockSize - blockCurrentSize) * sizeof(int64_t)); // write empty entries as well

                    int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 2 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) + sizeof(float) +
                                     3 * sizeof(int64_t) + blockSize * sizeof(int64_t);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            case COLUMN_FLOAT:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<float>& colFloat = dynamic_cast<const ColumnBase<float>&>(*(column.second));

                for (const auto& block : colFloat.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Float data with index = " << index;

                    auto data = block->GetData();
                    size_t blockCurrentSize = block->GetSize();
                    std::unique_ptr<float[]> emptyData(new float[blockSize - blockCurrentSize]);
                    std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                              std::numeric_limits<float>::max());
                    bool isCompressed = block->IsCompressed();
                    int32_t groupId = block->GetGroupId();
                    float min = block->GetMin();
                    float max = block->GetMax();
                    float avg = block->GetAvg();
                    float sum = block->GetSum();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }
                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
                    colDataFile.write(reinterpret_cast<char*>(&min), sizeof(float)); // write statistics min
                    colDataFile.write(reinterpret_cast<char*>(&max), sizeof(float)); // write statistics max
                    colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(float)); // write statistics sum
                    colDataFile.write(reinterpret_cast<const char*>(data),
                                      blockCurrentSize * sizeof(float)); // write block of data
                    colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                                      (blockSize - blockCurrentSize) * sizeof(float)); // write empty entries as well

                    int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 2 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) +
                                     4 * sizeof(float) + blockSize * sizeof(float);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            case COLUMN_DOUBLE:
            {
                uint32_t index = 0;
                uint64_t blockPosition = 0;

                const ColumnBase<double>& colDouble =
                    dynamic_cast<const ColumnBase<double>&>(*(column.second));

                for (const auto& block : colDouble.GetBlocksList())
                {
                    BOOST_LOG_TRIVIAL(debug)
                        << "Database: Saving block of Double data with index = " << index;

                    auto data = block->GetData();
                    size_t blockCurrentSize = block->GetSize();
                    std::unique_ptr<double[]> emptyData(new double[blockSize - blockCurrentSize]);
                    std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                              std::numeric_limits<double>::max());
                    bool isCompressed = block->IsCompressed();
                    int32_t groupId = block->GetGroupId();
                    double min = block->GetMin();
                    double max = block->GetMax();
                    float avg = block->GetAvg();
                    double sum = block->GetSum();

                    colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
                    colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
                    if (isNullable)
                    {
                        int32_t nullBitMaskLength =
                            (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                        colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                          sizeof(int32_t)); // write nullBitMask length
                        colDataFile.write(reinterpret_cast<char*>(block->GetNullBitmask()),
                                          nullBitMaskLength); // write nullBitMask
                    }
                    colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                                      sizeof(uint64_t)); // write block length (number of entries)
                    colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
                    colDataFile.write(reinterpret_cast<char*>(&min), sizeof(double)); // write statistics min
                    colDataFile.write(reinterpret_cast<char*>(&max), sizeof(double)); // write statistics max
                    colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
                    colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(double)); // write statistics sum
                    colDataFile.write(reinterpret_cast<const char*>(data),
                                      blockCurrentSize * sizeof(double)); // write block of data
                    colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                                      (blockSize - blockCurrentSize) * sizeof(double)); // write empty entries as well

                    int32_t nullBitMaskLength =
                        (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

                    colAddressFile.write(reinterpret_cast<char*>(&blockPosition), sizeof(uint64_t));

                    blockPosition += 2 * sizeof(int32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
                                     nullBitMaskLength * sizeof(char) + sizeof(bool) +
                                     sizeof(float) + 3 * sizeof(double) + blockSize * sizeof(double);
                    index += 1;

                    /* check if we did not get UINT32_MAX value in index - this value is reserved
                    to identify new block, which are just in memory and have never been persisted
                    into disk. If index reached this value, it means, the blockSize had been chosen
                    to too small value and we have reached our maximum number of blocks. No new
                    blocks will be persisted in order to at least save the current data.*/
                    if (index == UINT32_MAX)
                    {
                        BOOST_LOG_TRIVIAL(error)
                            << "ERROR: Database: When saving block of data into file: " << fileDataPath
                            << " tha maximum number of block has been reached. For that "
                               "reason, this block of data and data of other blocks whose have not "
                               "been persisted yet, will not be persisted in order to protect "
                               "already persisted data on disk.";
                        break;
                    }
                }
            }
            break;

            default:
                throw std::domain_error("Unsupported data type (when persisting database): " +
                                        std::to_string(type));
                break;
            }

            colAddressFile.close();
            colDataFile.close();
        }
        else
        {
            colDataFile.close();
            BOOST_LOG_TRIVIAL(error)
                << "Database: Could not open file " +
                       std::string(Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR + tableName +
                                   SEPARATOR + column.second->GetName() + COLUMN_ADDRESS_EXTENSION) +
                       " for writing. Persisting "
                << COLUMN_ADDRESS_EXTENSION
                << " file was not successful. Check if the process "
                   "have write access into the folder or file.";
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(error)
            << "ERROR: Database: WriteColumn, case3 - Could not open file " +
                   std::string(Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR +
                               tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION) +
                   " for writing. Persisting "
            << COLUMN_DATA_EXTENSION
            << " file was not successful. Check if the process "
               "have write access into the folder or file.";
    }
}
