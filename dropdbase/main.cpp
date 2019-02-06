#include <cstdio>
#include <iostream>
#include <chrono>
#include "Table.h"
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include "GpuSqlParser/MemoryStream.h"
#include <boost/log/core.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include "DatabaseGenerator.h"
#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "ColumnBase.h"
#include "Database.h"
#include "CSVDataImporter.h"

int main(int argc, char **argv)
{
	std::cout << "Start." << std::endl;
	if (argc > 1)
	{
		CSVDataImporter csvDataImporter("TargetLoc.csv");
		std::shared_ptr<Database> database = std::make_shared<Database>("TestDb", 1 << 24);
		Database::AddToInMemoryDatabaseList(database);
		std::cout << "Loading TargetLoc.csv ..." << std::endl;
		csvDataImporter.ImportTables(database);
		std::cout << "Saving dbs..." << std::endl;
		Database::SaveAllToDisk();
	}
	Database::LoadDatabasesFromDisk();
	if (Database::GetLoadedDatabases().size() == 0)
	{
		std::cout << "No dbs loaded, use some switch" << std::endl;
	}

	std::cout << "Parsing..." << std::endl;
	//GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();
	
    GpuSqlCustomParser parser(Database::GetDatabaseByName("TestDb"), "SELECT AVG(wealthIndexId), genderId FROM TargetLoc GROUP BY genderId;");
    parser.parse();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

    std::cout << "Elapsed time: " << elapsed.count()*1000 << " ms\n";

	system("PAUSE");
    return 0;
}