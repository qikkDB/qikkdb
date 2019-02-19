#include <cstdio>
#include <iostream>
#include <chrono>
#include <boost/log/core.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/trivial.hpp>
#include "QueryEngine/Context.h"
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include "DatabaseGenerator.h"
#include "Configuration.h"
#include "TCPServer.h"
#include "ClientPoolWorker.h"
#include "TCPClientHandler.h"
#include "ConsoleHandler.h"
#include "QueryEngine/GPUMemoryCache.h"
#include "CSVDataImporter.h"

int main(int argc, char **argv)
{
    boost::log::add_file_log("../log/ColmnarDB.log");

	Context::getInstance(); // Initialize CUDA context

	BOOST_LOG_TRIVIAL(info) << "Starting ColmnarDB...\n";
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::string> tableNames = { "TableA", "TableB" };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_DOUBLE}, {COLUMN_POLYGON}, {COLUMN_POINT}, {COLUMN_STRING}};
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("SimpleDb", 4, 1 << 5, true, tableNames, columnTypes);

	Database::AddToInMemoryDatabaseList(database);
	Database::SaveAllToDisk();

	Database::LoadDatabasesFromDisk();

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed(end - start);
	std::cout << "Elapsed time: " << elapsed.count() << " s." << std::endl;

	
	GpuSqlCustomParser parser(Database::GetDatabaseByName("SimpleDb"), "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 <= 20;");
	parser.parse()->PrintDebugString();

	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	RegisterCtrlCHandler(&tcpServer);
	tcpServer.Run();

	//Database::SaveAllToDisk();

	BOOST_LOG_TRIVIAL(info) << "Exiting cleanly...";
	
	/*
	std::vector<std::string> tableNames = { "TableA" };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 5, false, tableNames, columnTypes);


    GpuSqlCustomParser parser(database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 <= 20;");
    parser.parse()->PrintDebugString();

	std::cout << "Elapsed time: " << elapsed.count() << " s." << std::endl;
	*/

	return 0;
}
