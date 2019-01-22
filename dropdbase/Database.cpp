#include <filesystem>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdint>
#include <stdexcept>

#include "Database.h"
#include "Configuration.h"
#include "ColumnBase.h"
#include "Types/ComplexPolygon.pb.h"
#include "Table.h"

const std::shared_ptr<spdlog::logger>& Database::log_ = spdlog::get("root");

std::unordered_map<std::string, std::shared_ptr<Database>> Database::loadedDatabases_;

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
/// </summary>
/// <param name="databaseName">Database name.</param>
/// <param name="blockSize">Block size of all blocks in this database</param>
Database::Database(const char* databaseName, int32_t blockSize)
{
	name_ = databaseName;
	blockSize_ = blockSize;
}

Database::~Database()
{
}

void Database::Persist(const char* path)
{
	auto& tables = GetTables();
	auto& name = GetName();
	auto pathStr = std::string(path);

	log_->info("Saving database with name: '{}' and {} tables.", name, tables.size());

	if (std::filesystem::create_directory(path))
	{
		int32_t blockSize = GetBlockSize();
		int32_t tableSize = tables.size();

		//write file .db
		log_->debug("Saving .db file with name: '{}'", pathStr + name + ".db");
		std::ofstream dbFile(pathStr + name + ".db", std::ios::binary);

		int32_t dbNameLength = name.length();

		dbFile.write(reinterpret_cast<char*>(&dbNameLength), sizeof(int32_t)); //write db name length
		dbFile.write(name.c_str(), name.length()); //write db name
		dbFile.write(reinterpret_cast<char*>(&blockSize), sizeof(int32_t)); //write block size
		dbFile.write(reinterpret_cast<char*>(&tableSize), sizeof(int32_t)); //write number of tables
		for (auto& table : tables)
		{
			auto& columns = table.second.GetColumns();
			int32_t tableNameLength = table.first.length();
			int32_t columnNumber = columns.size();

			dbFile.write(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); //write table name length
			dbFile.write(table.first.c_str(), tableNameLength); //write table name
			dbFile.write(reinterpret_cast<char*>(&columnNumber), sizeof(int32_t)); //write number of columns of the table
			for (const auto& column : columns)
			{
				int32_t columnNameLength = column.first.length();

				dbFile.write(reinterpret_cast<char*>(&columnNameLength), sizeof(int32_t)); //write column name length
				dbFile.write(column.first.c_str(), columnNameLength); //write column name
			}
		}
		dbFile.close();

		//write files .col
		for (auto& table : tables)
		{
			auto& columns = table.second.GetColumns();

			for (const auto& column : columns)
			{
				log_->debug("Saving .col file with name: '{}'", pathStr + name + "_" + table.first + "_" + column.second->GetName() + ".col");

				std::ofstream colFile(pathStr + name + "_" + table.first + "_" + column.second->GetName() + ".col", std::ios::binary);

				int32_t type = column.second->GetColumnType();

				colFile.write(reinterpret_cast<char*>(&type), sizeof(int32_t)); //write type of column
				
				switch (type)
				{
				case COLUMN_POLYGON:
				{
					int32_t index = 0;

					const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon = dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

					for (const auto& block : colPolygon.GetBlocksList())
					{
						log_->debug("Saving block of ComplexPolygon data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								int32_t entryByteLength = entry.ByteSize();
								std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);

								entry.SerializeToArray(byteArray.get(), entryByteLength);

								colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
								colFile.write(byteArray.get(), entryByteLength); //write data of block
							}
							index += 1;
						}
					}
				}
					break;

				case COLUMN_POINT:
				{
					int32_t index = 0;

					const ColumnBase<ColmnarDB::Types::Point>& colPoint = dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

					for (const auto& block : colPoint.GetBlocksList())
					{
						log_->debug("Saving block of Point data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								int32_t entryByteLength = entry.ByteSize();
								std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);

								entry.SerializeToArray(byteArray.get(), entryByteLength);

								colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
								colFile.write(byteArray.get(), entryByteLength); //write data of block
							}
							index += 1;
						}
					}
				}
					break;

				case COLUMN_STRING:
				{
					int32_t index = 0;

					const ColumnBase<std::string>& colStr = dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

					for (const auto& block : colStr.GetBlocksList())
					{
						log_->debug("Saving block of String data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								int32_t entryByteLength = entry.length();

								colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
								colFile.write(entry.c_str(), entryByteLength); //write data of block
							}
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
						log_->debug("Saving block of Int32 data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								colFile.write(reinterpret_cast<const char*>(&entry), sizeof(int32_t)); //write data of block
							}
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
						log_->debug("Saving block of Int64 data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								colFile.write(reinterpret_cast<const char*>(&entry), sizeof(int64_t)); //write data of block
							}
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
						log_->debug("Saving block of Float data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								colFile.write(reinterpret_cast<const char*>(&entry), sizeof(float)); //write data of block
							}
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
						log_->debug("Saving block of Double data with index = {}.", index);

						auto& data = block->GetData();
						int32_t dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
							colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length
							for (const auto& entry : data)
							{
								colFile.write(reinterpret_cast<const char*>(&entry), sizeof(double)); //write data of block
							}
							index += 1;
						}
					}
				}
					break;

				default:
					throw std::exception("Unsupported data type (when persisting database).");
					break;
				}
			}
		}

		log_->info("Database '{}' was successfully saved to disc.", name);
	}
	else
	{
		log_->error("Failed to create directory when persisting database: '{}'.", name);
	}
}

/// <summary>
/// Save all databases currently in memory to disk
/// </summary>
/// <param name="path">Path to database storage directory</param>
void Database::SaveAllToDisk(const char * path)
{
	for (auto& database : Database::loadedDatabases_)
	{
		database.second->Persist(path);
	}
}

/// <summary>
/// Load databases from disk storage
/// </summary>
void Database::LoadDatabasesFromDisk()
{
	auto path = Configuration::DatabaseDir();

	if (std::filesystem::exists(path)) {
		for (auto& p : std::filesystem::directory_iterator(path))
		{
			auto extension = p.path().extension();
			if (extension == ".db")
			{
				auto database = Database::LoadDatabase(p.path().filename().stem().generic_string().c_str(), path);

				if (database != nullptr)
				{
					loadedDatabases_.insert( {database->name_, database} );
				}
			}
		}
	}
	else
	{
		log_->error("Directory {} does not exists.", path);
	}
}

/// <summary>
/// Load database from disc into memory.
/// </summary>
/// <param name="fileDbName">Name of the database file (*.db) without the ".db" suffix.</param>
/// <param name="path">Path to directory in which database files are.</param>
/// <returns>Shared pointer of database.</returns>
std::shared_ptr<Database> Database::LoadDatabase(const char* fileDbName, const char* path)
{
	log_->info("Loading database from directory: {} with file name: {}.", path, fileDbName);

	//read file .db
	std::ifstream dbFile(path + std::string(fileDbName) + ".db", std::ios::binary);
	
	int32_t dbNameLength;
	dbFile.read(reinterpret_cast<char*>(&dbNameLength), sizeof(int32_t)); //read db name length

	std::unique_ptr<char[]> dbName = std::make_unique<char[]>(dbNameLength);
	dbFile.read(dbName.get(), dbNameLength); //read db name

	int32_t blockSize;
	dbFile.read(reinterpret_cast<char*>(&blockSize), sizeof(int32_t)); //read block size

	int32_t tablesCount;
	dbFile.read(reinterpret_cast<char*>(&tablesCount), sizeof(int32_t)); //read number of tables

	std::shared_ptr<Database> database = std::make_shared<Database>(dbName.get(), blockSize);

	for (int32_t i = 0; i < tablesCount; i++)
	{
		int32_t tableNameLength;
		dbFile.read(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); //read db name length

		std::unique_ptr<char[]> tableName = std::make_unique<char[]>(tableNameLength);
		dbFile.read(tableName.get(), tableNameLength); //read db name

		database->tables_.insert( {tableName.get(), Table (database, tableName.get())} );

		int32_t columnCount;
		dbFile.read(reinterpret_cast<char*>(&columnCount), sizeof(int32_t)); //read number of columns

		std::vector<std::string> columnNames;

		for (int32_t j = 0; j < columnCount; j++)
		{
			int32_t columnNameLength;
			dbFile.read(reinterpret_cast<char*>(&columnNameLength), sizeof(int32_t)); //read column name length

			std::unique_ptr<char[]> columnName = std::make_unique<char[]>(columnNameLength);
			dbFile.read(columnName.get(), columnNameLength); //read column name

			columnNames.push_back(columnName.get());
		}

		auto& table = database->tables_.at(tableName.get());
		LoadColumns(path, dbName.get(), table, columnNames); //read files .col
	}

	return std::shared_ptr<Database>();
}

/// <summary>
/// Load columns of a table into memory from disc.
/// </summary>
/// <param name="path">Path directory, where column files (*.col) are.</param>
/// <param name="table">Instance of table into which the columns should be added.</param>
/// <param name="columnNames">Names of particular columns.</param>
void Database::LoadColumns(const char* path, const char* dbName, Table& table, const std::vector<std::string>& columnNames)
{
	//TODO dorobit funkcionalitu !!!!!!!!!!!!!!!!
}

void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database> database)
{
	loadedDatabases_.insert({ database->name_, database });
}


