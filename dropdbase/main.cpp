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

	std::vector<std::string> tableNames = { "TableA" };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 5, false, tableNames, columnTypes);
	std::vector<std::string> tableNames1 = { {"TableAA"},{"TableBB"},{"TableCC"},{"TableDD"} };
	std::vector<DataType> columnTypes2 = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database2 = DatabaseGenerator::GenerateDatabase("TestDb2", 2, 1 << 5, false, tableNames1, columnTypes2);
	std::shared_ptr<Database> database3 = DatabaseGenerator::GenerateDatabase("VeninkinaDB", 2, 1 << 5, false, tableNames, columnTypes);
	Database::AddToInMemoryDatabaseList(database);
	Database::AddToInMemoryDatabaseList(database2);
	Database::AddToInMemoryDatabaseList(database3);

	//GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();

	GpuSqlCustomParser parser(database, "SHOW TABLES FROM TestDb2;");
	parser.parse();

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed(end - start);

	std::cout << "Elapsed time: " << elapsed.count() << " s\n";


	return 0;
}
