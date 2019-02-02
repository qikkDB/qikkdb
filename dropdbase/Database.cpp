#include <boost/filesystem.hpp>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <boost/log/trivial.hpp>

#include "Database.h"
#include "Configuration.h"
#include "ColumnBase.h"
#include "Types/ComplexPolygon.pb.h"
#include "Table.h"

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

/// <summary>
/// Save database from memory to disk.
/// </summary>
/// <param name="path">Path to database storage directory</param>
void Database::Persist(const char* path)
{
	auto& tables = GetTables();
	auto& name = GetName();
	auto pathStr = std::string(path);

	BOOST_LOG_TRIVIAL(info) << "Saving database with name: " << name << " and " << tables.size() << " tables." << std::endl;

	boost::filesystem::create_directories(path);
	
	int32_t blockSize = GetBlockSize();
	int32_t tableSize = tables.size();

	//write file .db
	BOOST_LOG_TRIVIAL(debug) << "Saving .db file with name: " << pathStr << name << " .db" << std::endl;
	std::ofstream dbFile(pathStr + name + ".db", std::ios::binary);

	int32_t dbNameLength = name.length() + 1; // +1 because '\0'

	dbFile.write(reinterpret_cast<char*>(&dbNameLength), sizeof(int32_t)); //write db name length
	dbFile.write(name.c_str(), dbNameLength); //write db name
	dbFile.write(reinterpret_cast<char*>(&blockSize), sizeof(int32_t)); //write block size
	dbFile.write(reinterpret_cast<char*>(&tableSize), sizeof(int32_t)); //write number of tables
	for (auto& table : tables)
	{
		auto& columns = table.second.GetColumns();
		int32_t tableNameLength = table.first.length() + 1; // +1 because '\0'
		int32_t columnNumber = columns.size();

		dbFile.write(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); //write table name length
		dbFile.write(table.first.c_str(), tableNameLength); //write table name
		dbFile.write(reinterpret_cast<char*>(&columnNumber), sizeof(int32_t)); //write number of columns of the table
		for (const auto& column : columns)
		{
			int32_t columnNameLength = column.first.length() + 1; // +1 because '\0'

			dbFile.write(reinterpret_cast<char*>(&columnNameLength), sizeof(int32_t)); //write column name length
			dbFile.write(column.first.c_str(), columnNameLength); //write column name
		}
	}
	dbFile.close();

	//write files .col:
	for (auto& table : tables)
	{
		auto& columns = table.second.GetColumns();

		for (const auto& column : columns)
		{
			BOOST_LOG_TRIVIAL(debug) << "Saving .col file with name: " << pathStr << name << "_" << table.first << "_" << column.second->GetName() << " .col" << std::endl;

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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of ComplexPolygon data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							int32_t entryByteLength = entry.ByteSize();
							std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);

							entry.SerializeToArray(byteArray.get(), entryByteLength);

							colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
							colFile.write(byteArray.get(), entryByteLength); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of Point data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							int32_t entryByteLength = entry.ByteSize();
							std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);

							entry.SerializeToArray(byteArray.get(), entryByteLength);

							colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
							colFile.write(byteArray.get(), entryByteLength); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of String data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							int32_t entryByteLength = entry.length() + 1; // +1 because '\0'

							colFile.write(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //write entry length
							colFile.write(entry.c_str(), entryByteLength); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of Int32 data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							colFile.write(reinterpret_cast<const char*>(&entry), sizeof(int32_t)); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of Int64 data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							colFile.write(reinterpret_cast<const char*>(&entry), sizeof(int64_t)); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of Float data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							colFile.write(reinterpret_cast<const char*>(&entry), sizeof(float)); //write entry data
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
					BOOST_LOG_TRIVIAL(debug) << "Saving block of Double data with index = " << index << "." << std::endl;

					auto& data = block->GetData();
					int32_t dataLength = data.size();

					if (data.size() > 0)
					{
						colFile.write(reinterpret_cast<char*>(&index), sizeof(int32_t)); //write index
						colFile.write(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //write block length (number of entries)
						for (const auto& entry : data)
						{
							colFile.write(reinterpret_cast<const char*>(&entry), sizeof(double)); //write entry data
						}
						index += 1;
					}
				}
			}
				break;

			default:
				throw std::domain_error("Unsupported data type (when persisting database).");
				break;
			}

			colFile.close();
		}
	}

	BOOST_LOG_TRIVIAL(info) << "Database " << name << " was successfully saved to disc." << std::endl;
}

/// <summary>
/// Save all databases currently in memory to disk. All databases will be saved in the same directory
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
/// Load databases from disk storage. Databases .db and .col files have to be in the same directory, so all databases have to be in the same dorectory to be loaded using this procedure
/// </summary>
void Database::LoadDatabasesFromDisk()
{
	auto &path = Configuration::GetInstance().GetDatabaseDir();

	if (boost::filesystem::exists(path)) {
		for (auto& p : boost::filesystem::directory_iterator(path))
		{
			auto extension = p.path().extension();
			if (extension == ".db")
			{
				auto database = Database::LoadDatabase(p.path().filename().stem().generic_string().c_str(), path.c_str());

				if (database != nullptr)
				{
					loadedDatabases_.insert( {database->name_, database} );
				}
			}
		}
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Directory " << path << " does not exists." << std::endl;
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
	BOOST_LOG_TRIVIAL(info) << "Loading database from: " << path << fileDbName << ".db." << std::endl;

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
		dbFile.read(reinterpret_cast<char*>(&tableNameLength), sizeof(int32_t)); //read table name length

		std::unique_ptr<char[]> tableName = std::make_unique<char[]>(tableNameLength);
		dbFile.read(tableName.get(), tableNameLength); //read table name

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

	dbFile.close();

	return database;
}

/// <summary>
/// Load columns of a table into memory from disc.
/// </summary>
/// <param name="path">Path directory, where column files (*.col) are.</param>
/// <param name="table">Instance of table into which the columns should be added.</param>
/// <param name="columnNames">Names of particular columns.</param>
void Database::LoadColumns(const char* path, const char* dbName, Table& table, const std::vector<std::string>& columnNames)
{
	for (const std::string& columnName : columnNames)
	{
		//read files .col:
		std::string pathStr = std::string(path);

		BOOST_LOG_TRIVIAL(info) << "Loading .col file with name: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col." << std::endl;

		std::ifstream colFile(pathStr + dbName + "_" + table.GetName() + "_" + columnName + ".col", std::ios::binary);

		int32_t nullIndex = 0;

		int32_t type;
		colFile.read(reinterpret_cast<char*>(&type), sizeof(int32_t)); //read type of column

		switch (type)
		{
			case COLUMN_POLYGON:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_POLYGON);

				auto& columnPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnPolygon.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty ComplexPolygon block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;

						for (int32_t j = 0; j < dataLength; j++)
						{
							int32_t entryByteLength;
							colFile.read(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //read entry length

							std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);
							colFile.read(byteArray.get(), entryByteLength); //read entry data

							ColmnarDB::Types::ComplexPolygon entryDataPolygon;

							entryDataPolygon.ParseFromArray(byteArray.get(), entryByteLength);

							dataPolygon.push_back(entryDataPolygon);
						}

						columnPolygon.AddBlock(dataPolygon);
						BOOST_LOG_TRIVIAL(debug) << "Added ComplexPolygon block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
			break;

			case COLUMN_POINT:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_POINT);

				auto& columnPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnPoint.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty Point block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<ColmnarDB::Types::Point> dataPoint;

						for (int32_t j = 0; j < dataLength; j++)
						{
							int32_t entryByteLength;
							colFile.read(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //read entry length

							std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);
							colFile.read(byteArray.get(), entryByteLength); //read entry data

							ColmnarDB::Types::Point entryDataPoint;

							entryDataPoint.ParseFromArray(byteArray.get(), entryByteLength);

							dataPoint.push_back(entryDataPoint);
						}

						columnPoint.AddBlock(dataPoint);
						BOOST_LOG_TRIVIAL(debug) << "Added Point block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
			break;

			case COLUMN_STRING:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_STRING);

				auto& columnString = dynamic_cast<ColumnBase<std::string>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnString.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty String block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<std::string> dataString;

						for (int32_t j = 0; j < dataLength; j++)
						{
							int32_t entryByteLength;
							colFile.read(reinterpret_cast<char*>(&entryByteLength), sizeof(int32_t)); //read entry length

							std::unique_ptr<char[]> byteArray = std::make_unique<char[]>(entryByteLength);
							colFile.read(byteArray.get(), entryByteLength); //read entry data

							std::string entryDataString(byteArray.get());
							dataString.push_back(entryDataString);
						}

						columnString.AddBlock(dataString);
						BOOST_LOG_TRIVIAL(debug) << "Added String block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
				break;

			case COLUMN_INT:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_INT);

				auto& columnInt = dynamic_cast<ColumnBase<int32_t>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index
					
					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnInt.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty Int32 block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<int32_t> dataInt;

						for (int32_t j = 0; j < dataLength; j++)
						{
							int32_t entryDataInt;
							colFile.read(reinterpret_cast<char*>(&entryDataInt), sizeof(int32_t)); //read entry data

							dataInt.push_back(entryDataInt);
						}

						columnInt.AddBlock(dataInt);
						BOOST_LOG_TRIVIAL(debug) << "Added Int32 block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
				break;

			case COLUMN_LONG:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_LONG);

				auto& columnLong = dynamic_cast<ColumnBase<int64_t>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnLong.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty Int64 block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<int64_t> dataLong;

						for (int32_t j = 0; j < dataLength; j++)
						{
							int64_t entryDataLong;
							colFile.read(reinterpret_cast<char*>(&entryDataLong), sizeof(int64_t)); //read entry data

							dataLong.push_back(entryDataLong);
						}

						columnLong.AddBlock(dataLong);
						BOOST_LOG_TRIVIAL(debug) << "Added Int64 block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
			break;

			case COLUMN_FLOAT:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_FLOAT);

				auto& columnFloat = dynamic_cast<ColumnBase<float>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnFloat.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty Float block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<float> dataFloat;

						for (int32_t j = 0; j < dataLength; j++)
						{
							float entryDataFloat;
							colFile.read(reinterpret_cast<char*>(&entryDataFloat), sizeof(float)); //read entry data

							dataFloat.push_back(entryDataFloat);
						}

						columnFloat.AddBlock(dataFloat);
						BOOST_LOG_TRIVIAL(debug) << "Added Float block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
				break;

			case COLUMN_DOUBLE:
			{
				table.CreateColumn(columnName.c_str(), COLUMN_DOUBLE);

				auto& columnDouble = dynamic_cast<ColumnBase<double>&>(*table.GetColumns().at(columnName));

				while (!colFile.eof())
				{
					int32_t index;
					colFile.read(reinterpret_cast<char*>(&index), sizeof(int32_t)); //read block index

					//this is needed because of how EOF is checked:
					if (colFile.eof())
					{
						BOOST_LOG_TRIVIAL(debug) << "Loading of the file: " << pathStr + dbName << "_" << table.GetName() << "_" << columnName << ".col has finished successfully." << std::endl;
						break;
					}

					int32_t dataLength;
					colFile.read(reinterpret_cast<char*>(&dataLength), sizeof(int32_t)); //read data length (number of entries)

					if (index != nullIndex) //there is null block
					{
						columnDouble.AddBlock(); //add empty block
						BOOST_LOG_TRIVIAL(debug) << "Added empty Double block at index: " << nullIndex << "." << std::endl;
					}
					else //read data from block
					{
						std::vector<double> dataDouble;

						for (int32_t j = 0; j < dataLength; j++)
						{
							double entryDataDouble;
							colFile.read(reinterpret_cast<char*>(&entryDataDouble), sizeof(double)); //read entry data

							dataDouble.push_back(entryDataDouble);
						}

						columnDouble.AddBlock(dataDouble);
						BOOST_LOG_TRIVIAL(debug) << "Added Double block with data at index: " << index << "." << std::endl;
					}

					nullIndex += 1;
				}
			}
				break;

			default:
				throw std::domain_error("Unsupported data type (when loading database).");
		}

		colFile.close();
	}
}

/// <summary>
/// Creates table with given name and columns and adds it to database. If the table already existed, create missing columns if there are any missing.
/// </summary>
/// <returns>Newly created table</returns>
/// <param name="columns">Columns with types.</param>
/// <param name="tableName">Table name.</param>
Table& Database::CreateTable(const std::unordered_map<std::string, DataType>& columns, const char* tableName)
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
					throw std::domain_error("Column type in CreateTable does not match with existing column.");
				}
			}
			else
			{
				table.CreateColumn(entry.first.c_str(), entry.second);
			}
		}

		return table;
	}
	else
	{
		tables_.insert({ tableName,Table(Database::GetDatabaseByName(name_), tableName) });
		auto& table = tables_.at(tableName);

		for (auto& entry : columns)
		{
			table.CreateColumn(entry.first.c_str(), entry.second);
		}

		return table;
	}
}

/// <summary>
/// Add database to in memory list
/// </summary>
/// <param name="database">Database to add</param>
void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database> database)
{
	loadedDatabases_.insert({ database->name_, database });
}

/// <summary>
/// Get number of blocks
/// </summary>
/// <returns>Number of blocks</param>
int Database::GetBlockCount()
{
	for (auto& table : tables_) 
	{
		for (auto &column : table.second.GetColumns()) 
		{
			return column.second.get()->GetBlockCount();
		}
	}
	return 0;
}

