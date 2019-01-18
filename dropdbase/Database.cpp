#include <filesystem>
#include <fstream>

#include "Database.h"

std::unordered_map<std::string, std::shared_ptr<Database>> Database::loadedDatabases_;

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
/// </summary>
/// <param name="databaseName">Database name.</param>
/// <param name="blockSize">Block size of all blocks in this database</param>
Database::Database(std::string databaseName, int blockSize) : log_(spdlog::get("root"))
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
		//write file .db
		log_->debug("Saving .db file with name: '{}'", pathStr + name + ".db");
		std::ofstream dbFile(pathStr + name + ".db", std::ios::binary);
		dbFile.write(name.c_str(), name.length); //argumetns: buffer, size
		dbFile.write((char*) GetBlockSize(), sizeof(unsigned int));
		dbFile.write((char*) tables.size(), sizeof(unsigned int));
		for (auto table : tables)
		{
			auto columns = table.second.GetColumns();
			dbFile.write(table.first.c_str(), sizeof(table.first.c_str())); //write table name
			dbFile.write(columns.size(), sizeof(unsigned int)); //number of columns of the table
			for (auto column : columns)
			{
				dbFile.write(column.first.c_str(), sizeof(column.first.c_str())); //name of the column
			}
		}
		dbFile.close();

		//write files .col
		for (auto table : tables)
		{
			auto columns = table.second.GetColumns();

			for (auto column : columns)
			{
				log_->debug("Saving .col file with name: '{}'", pathStr + name + "_" + table.first + "_" + column.GetName() + ".col");

				std::ofstream colFile(pathStr + name + "_" + table.first + "_" + column.GetName() + ".col", std::ios::binary);

				auto type = std::string(column.GetColumnType());

				colFile.write(type.c_str(), sizeof(type.c_str())); //type of column
				
				unsigned int index = 0;
				for (auto block : column.GetBlocks())
				{
					//save block of type - ComplexPolygon:
					if (type == "ComplexPolygon")
					{
						log_->debug("Saving block of ComplexPolygon data with index = {}.", index);

						auto dataStr = block.GetData();

						if (dataStr != nullptr)
						{
							colFile.write((char*)index, sizeof(unsigned int)); //write index
							colFile.write((char*)dataStr.length, sizeof(unsigned int)); //write block length
							for (auto entry : dataStr) //write data of block
							{
								colFile.write((char*)entry.c_str, entry.length);
							}
							index += 1;
						}
					}
					//save block of type - Point:
					else if (type == "Point")
					{
						log_->debug("Saving block of Point data with index = {}.", index);

						auto dataStr = block.GetData();

						if (dataStr != nullptr)
						{
							colFile.write((char*)index, sizeof(unsigned int)); //write index
							colFile.write((char*)dataStr.length, sizeof(unsigned int)); //write block length
							for (auto entry : dataStr) //write data of block
							{
								colFile.write((char*)entry.c_str, entry.length);
							}
							index += 1;
						}
					}
					//TODO add writing of other types as well !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database> database)
{

	loadedDatabases_.insert({ database->name_, database });
}


