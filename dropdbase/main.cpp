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

int main(int argc, char **argv)
{
	/*
	Context::getInstance(); // Initialize CUDA context

    boost::log::add_file_log("../log/ColmnarDB.log");
    boost::log::add_console_log(std::cout);
	BOOST_LOG_TRIVIAL(info) << "Starting ColmnarDB...\n";
	Database::LoadDatabasesFromDisk();
	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	tcpServer.Run();
	Database::SaveAllToDisk(Configuration::GetInstance().GetDatabaseDir().c_str());
	*/

	
	std::vector<std::string> tableNames = { "TableA" };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_INT} /*, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} */ };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, 1 << 24, false, tableNames, columnTypes);

	//GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();
	

    GpuSqlCustomParser parser(database, "SELECT colInteger2 * 2, COUNT(colInteger1) / -2 FROM TableA GROUP BY colInteger2;");
    parser.parse();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	
	return 0;
}