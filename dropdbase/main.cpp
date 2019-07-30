/// \mainpage Project summary
/// SQL-like database application with query executing on GPU.
/// <br />
/// <b>Used programming language:</b>
///   - C++
///
/// <b>Used technologies:</b>
///   - CUDA
///   - Antlr
///   - Google Protocol Buffers

#include "CSVDataImporter.h"
#include <cstdio>
#include <iostream>
#include <chrono>
#include <boost/log/trivial.hpp>
// TODO solve undefined references
//#include <boost/log/utility/setup/from_stream.hpp>
#include "QueryEngine/Context.h" 
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include "DatabaseGenerator.h"
#include "Configuration.h"
#include "TCPServer.h"
#include "ClientPoolWorker.h"
#include "TCPClientHandler.h"
#include "ConsoleHandler.h"
#include "QueryEngine/GPUMemoryCache.h"

/// Startup function, called automatically.
/// <param name="argc">program argument count</param>
/// <param name="argv">program arguments (for CSV importing): [csv-path [new-db-name]]</param>
/// <returns>Exit code (0 - OK)</returns>
int main(int argc, char **argv)
{
	/*
	//TODO solve undefined references
	std::ifstream logConfigFile("../configuration/log_config");
	if (logConfigFile.fail())
	{
		logConfigFile = std::ifstream("../configuration/log_config.default");
	}
	boost::log::init_from_stream(logConfigFile);
	*/

	//std::string dbName = "TestDatabase";
	//std::vector<std::string> tableNames = { "TestTable1" };
	//std::vector<DataType> columnTypes = { {COLUMN_INT},
	//	 {COLUMN_INT},
	//	 {COLUMN_LONG},
	//	 {COLUMN_LONG},
	//	 {COLUMN_LONG},
	//	 {COLUMN_FLOAT},
	//	 {COLUMN_FLOAT},
	//	 {COLUMN_DOUBLE},
	//	 {COLUMN_DOUBLE},
	//	 {COLUMN_POLYGON},
	//	 {COLUMN_POINT} };
	//std::shared_ptr<Database> compressionDatabase = DatabaseGenerator::GenerateDatabase(dbName.c_str(), 2, 1<<18, false, tableNames, columnTypes);
	//Database::AddToInMemoryDatabaseList(compressionDatabase);
	//Database::SaveAllToDisk();
	//return 0;

	BOOST_LOG_TRIVIAL(info) << "Starting TellStoryDB...";
	Context::getInstance(); // Initialize CUDA context

	// Import CSV file if entered as program argument
	if (argc > 1)
	{
		CSVDataImporter csvDataImporter(argv[1]);
		////CSVDataImporter csvDataImporter(R"(D:\DataGenerator\output\TargetLoc1B.csv)");
		std::shared_ptr<Database> database = std::make_shared<Database>(argc > 2 ? argv[2] : "TestDb", 1048576);
		Database::AddToInMemoryDatabaseList(database);
		BOOST_LOG_TRIVIAL(info) << "Loading CSV from \"" << argv[1] << "\"";
		csvDataImporter.ImportTables(database);
		Database::SaveAllToDisk();
		for (auto& db : Database::GetDatabaseNames())
		{
			Database::RemoveFromInMemoryDatabaseList(db.c_str());
		}
		return 0;
	}

	BOOST_LOG_TRIVIAL(info) << "Loading databases...";
	Database::LoadDatabasesFromDisk();
	BOOST_LOG_TRIVIAL(info) << "All databases loaded.";

	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	RegisterCtrlCHandler(&tcpServer);
	tcpServer.Run();

	Database::SaveAllToDisk();
	BOOST_LOG_TRIVIAL(info) << "Exiting cleanly...";

	/*CSVDataImporter csvDataImporter(R"(D:\testing-data\TargetLoc100M.csv)");
	std::shared_ptr<Database> database = std::make_shared<Database>("TestDb", 100000000);
	Database::AddToInMemoryDatabaseList(database);
	BOOST_LOG_TRIVIAL(info) << "Loading TargetLoc.csv ...";
	csvDataImporter.ImportTables(database);*/
	/*
	for (int i = 0; i < 2; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();

		GpuSqlCustomParser parser(Database::GetDatabaseByName("TestDb"), "SELECT ageId, COUNT(ageId) FROM TargetLoc1B WHERE latitude > 48.163267512773274 AND latitude < 48.17608989851882 AND longitude > 17.19991468973717 AND longitude < 17.221200700479358 GROUP BY ageId;");
		parser.parse();// ->PrintDebugString();

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed(end - start);
		BOOST_LOG_TRIVIAL(info) << "Elapsed time: " << elapsed.count() << " s.";
	}
	*/

	for (auto& db : Database::GetDatabaseNames())
	{
		Database::RemoveFromInMemoryDatabaseList(db.c_str());
	}
	return 0;
}
