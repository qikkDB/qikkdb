#include "CSVDataImporter.h"
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


int main(int argc, char **argv)
{
    boost::log::add_file_log("../log/ColmnarDB.log");
    boost::log::add_console_log(std::cout);

	Context::getInstance(); // Initialize CUDA context

	BOOST_LOG_TRIVIAL(info) << "Starting ColmnarDB...\n";
	Database::LoadDatabasesFromDisk();
	
	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	RegisterCtrlCHandler(&tcpServer);
	tcpServer.Run();

	Database::SaveAllToDisk();
	BOOST_LOG_TRIVIAL(info) << "Exiting cleanly...";
	
	/*CSVDataImporter csvDataImporter(R"(D:\testing-data\TargetLoc100M.csv)");
	std::shared_ptr<Database> database = std::make_shared<Database>("TestDb", 100000000);
	Database::AddToInMemoryDatabaseList(database);
	std::cout << "Loading TargetLoc.csv ..." << std::endl;
	csvDataImporter.ImportTables(database);*/
    /*
	for (int i = 0; i < 2; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();

		GpuSqlCustomParser parser(Database::GetDatabaseByName("TestDb"), "SELECT ageId, COUNT(ageId) FROM TargetLoc1B WHERE latitude > 48.163267512773274 AND latitude < 48.17608989851882 AND longitude > 17.19991468973717 AND longitude < 17.221200700479358 GROUP BY ageId;");
		parser.parse();// ->PrintDebugString();

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed(end - start);
		std::cout << "Elapsed time: " << elapsed.count() << " s." << std::endl;
	}
	*/
	for (auto& db : Database::GetDatabaseNames())
	{
		Database::RemoveFromInMemoryDatabaseList(db.c_str());
	}
	return 0;
}
