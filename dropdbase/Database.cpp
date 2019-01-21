#include <filesystem>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "Database.h"
#include "Configuration.h"
#include "ColumnBase.h"

std::unordered_map<std::string, std::shared_ptr<Database>> Database::loadedDatabases_;

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
/// </summary>
/// <param name="databaseName">Database name.</param>
/// <param name="blockSize">Block size of all blocks in this database</param>
Database::Database(const char* databaseName, int blockSize) : log_(spdlog::get("root"))
{
	name_ = databaseName;
	blockSize_ = blockSize;
}

Database::~Database()
{
}

void Database::Persist(const char * path)
{
	auto tables = GetTables();
	auto name = GetName();
	auto pathStr = std::string(path);

	log_->info("Saving database with name: '{}' and {} tables.", name, tables.size());

	if (std::filesystem::create_directory(path))
	{
		int blockSize = GetBlockSize();
		int tableSize = tables.size();

		//write file .db
		log_->debug("Saving .db file with name: '{}'", pathStr + name + ".db");
		std::ofstream dbFile(pathStr + name + ".db", std::ios::binary);

		int dbNameLength = name.length();

		dbFile.write(reinterpret_cast<char*>(&dbNameLength), sizeof(int)); //write db name length
		dbFile.write(name.c_str(), name.length()); //write db name
		dbFile.write(reinterpret_cast<char*>(&blockSize), sizeof(int)); //write block size
		dbFile.write(reinterpret_cast<char*>(&tableSize), sizeof(int)); //write number of tables
		for (auto& table : tables)
		{
			auto columns = table.second.GetColumns();
			int tableNameLength = table.first.length();

			dbFile.write(reinterpret_cast<char*>(&tableNameLength), sizeof(int)); //write table name length
			dbFile.write(table.first.c_str(), tableNameLength); //write table name
			dbFile.write(reinterpret_cast<char*>(columns.size()), sizeof(int)); //write number of columns of the table
			for (const auto& column : columns)
			{
				int columnNameLength = column.first.length();

				dbFile.write(reinterpret_cast<char*>(&columnNameLength), sizeof(int)); //write column name length
				dbFile.write(column.first.c_str(), columnNameLength); //write column name
			}
		}
		dbFile.close();

		//write files .col
		for (auto& table : tables)
		{
			auto columns = table.second.GetColumns();

			for (const auto& column : columns)
			{
				log_->debug("Saving .col file with name: '{}'", pathStr + name + "_" + table.first + "_" + column.second->GetName() + ".col");

				std::ofstream colFile(pathStr + name + "_" + table.first + "_" + column.second->GetName() + ".col", std::ios::binary);

				int type = column.second->GetColumnType();

				colFile.write(reinterpret_cast<char*>(&type), sizeof(int)); //write type of column
				
				switch (type)
				{
				case COLUMN_INT:
					unsigned int index = 0;

					const ColumnBase<int>* colInt = dynamic_cast<const ColumnBase<int>*>(column.second.get());
					
					for (const auto& block : colInt->GetBlocksList())
					{
						log_->debug("Saving block of Integer data with index = {}.", index);

						auto& data = block->GetData();
						int dataLength = data.size();

						if (data.size() > 0)
						{
							colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
							colFile.write(reinterpret_cast<char*>(dataLength), sizeof(unsigned int)); //write block length
							for (auto entry : data)
							{
								colFile.write(reinterpret_cast<char*>(&entry), sizeof(int)); //write data of block
							}
							index += 1;
						}
					}

					break;

					//TODO tu pojdu vsetky cases
				}

				//for (auto block : column.GetBlocks())
				//{
				//	//save block of type - String
				//	if (type == "String")
				//	{
				//		log_->debug("Saving block of Point data with index = {}.", index);

				//		auto dataStr = block.GetData();

				//		if (dataStr != nullptr)
				//		{
				//			int strLength = dataStr.length();

				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			dbFile.write(reinterpret_cast<char*>(&strLength), sizeof(int)); //write string length
				//			colFile.write(static_cast<char*>(dataStr.length), strLength); //write block length
				//			for (auto entry : dataStr) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry.c_str()), entry.length);
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - ComplexPolygon:
				//	else if (type == "ComplexPolygon")
				//	{
				//		log_->debug("Saving block of ComplexPolygon data with index = {}.", index);

				//		auto dataStr = block.GetData();

				//		if (dataStr != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(dataStr.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : dataStr) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry.c_str()), entry.length);
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - Point:
				//	else if (type == "Point")
				//	{
				//		log_->debug("Saving block of Point data with index = {}.", index);

				//		auto dataStr = block.GetData();

				//		if (dataStr != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(dataStr.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : dataStr) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry.c_str()), entry.length);
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - Integer:
				//	else if (type == "Int")
				//	{
				//		log_->debug("Saving block of Integer data with index = {}.", index);

				//		auto data = block.GetData();

				//		if (data != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(data.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : data) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry), sizeof(int));
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - Long:
				//	else if (type == "Long")
				//	{
				//		log_->debug("Saving block of Long data with index = {}.", index);

				//		auto data = block.GetData();

				//		if (data != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(data.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : data) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry), sizeof(long));
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - Double:
				//	else if (type == "Double")
				//	{
				//		log_->debug("Saving block of Double data with index = {}.", index);

				//		auto data = block.GetData();

				//		if (data != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(data.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : data) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry), sizeof(double));
				//			}
				//			index += 1;
				//		}
				//	}
				//	//save block of type - Float:
				//	else if (type == "Float")
				//	{
				//		log_->debug("Saving block of Float data with index = {}.", index);

				//		auto data = block.GetData();

				//		if (data != nullptr)
				//		{
				//			colFile.write(reinterpret_cast<char*>(&index), sizeof(unsigned int)); //write index
				//			colFile.write(static_cast<char*>(data.length), sizeof(unsigned int)); //write block length
				//			for (auto entry : data) //write data of block
				//			{
				//				colFile.write(static_cast<char*>(entry), sizeof(float));
				//			}
				//			index += 1;
				//		}
				//	}
				//}
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
	for (std::pair<std::string, std::shared_ptr<Database>> database : Database::loadedDatabases_)
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

std::shared_ptr<Database> Database::LoadDatabase(const char * fileDbName, const char * path)
{
	log_->info("Loading database from directory: {} with file name: {}.", path, fileDbName);

	//read file .db
	std::ifstream dbFile(path + std::string(fileDbName) + ".db", std::ios::binary);
	
	std::unique_ptr<char[]> dbName = std::make_unique<char[]>(length); //dynamic allocation
	dbFile.read(dbName, 5);
	auto blockSize = dbFile.get();
	auto tablesCount = dbFile.get();

	Database database(dbName, blockSize);

	for (auto i = 0; i < tablesCount; i++)
	{
		//TODO dorobit !!!!!!!!!!!!!!
	}

	return std::shared_ptr<Database>();
}

void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database> database)
{

	loadedDatabases_.insert({ database->name_, database });
}


