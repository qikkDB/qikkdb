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
#include "ColumnBase.h"
#include "Database.h"

int main(int argc, char **argv)
{
	Context::getInstance(); // Initialize CUDA context
	
	boost::log::add_file_log("../log/ColmnarDB.log");
	boost::log::add_console_log(std::cout);

	int blockSize = 1 << 5;

	std::vector<std::string> tableNames = { {"TableA"} };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, blockSize, true, tableNames, columnTypes);
	Database::AddToInMemoryDatabaseList(database);

	GpuSqlCustomParser parser(database, "INSERT INTO TableB (colPolygon1, colPoint1) VALUES (POLYGON((20 15, 11 12, 20 15),(21 30, 35 36, 30 20, 21 30),(61 80,90 89,112 110, 61 80)), POINT(2 5));");


	//GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();

	//GpuSqlCustomParser parser(database, "INSERT INTO TableA (colInteger1, colPolygon1) VALUES (2, POLYGON((10 11, 11 12, 10 11),(21 30, 35 36, 30 20, 21 30),(61 80,90 89,112 110, 61 80)));");
	parser.parse()->PrintDebugString();

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed(end - start);

	std::cout << "Elapsed time: " << elapsed.count() << " s\n";


	return 0;
}
