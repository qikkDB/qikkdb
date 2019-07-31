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
/// <param name="argc">not used parameter</param>
/// <param name="argv">not used parameter</param>
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

	Context::getInstance();

	std::cout << "Importing data has started..." << std::endl;

	//CSVDataImporter csvDataImporter1(R"(../../data/GeoPoint.csv)");
	//std::shared_ptr<Database> database1 = std::make_shared<Database>("GeoTest", 131072);
	//Database::AddToInMemoryDatabaseList(database1);
	//std::cout << "Loading GeoPoint.csv ..." << std::endl;
	//csvDataImporter1.ImportTables(database1);

	//CSVDataImporter csvDataImporter2(R"(../../data/TargetLoc1B.csv)");
	//std::shared_ptr<Database> database2 = std::make_shared<Database>("TargetLocator", 134217728);
	//Database::AddToInMemoryDatabaseList(database2);
	//std::cout << "Loading TargetLoc1B.csv ..." << std::endl;
	//csvDataImporter2.ImportTables(database2);

	//Database::SaveAllToDisk();
	//return 0;

	CSVDataImporter csvDataImporter3(R"(../../data/trimmed-trips-part1.csv)");
	const std::vector<DataType> types{
		COLUMN_STRING,
		COLUMN_LONG,
		COLUMN_LONG,
		COLUMN_INT,
		COLUMN_DOUBLE,
		COLUMN_DOUBLE,
		COLUMN_STRING
	};
	const std::string tableName = "trips";
	csvDataImporter3.SetTypes(types);
	csvDataImporter3.SetTableName(tableName);
	std::shared_ptr<Database> database3 = std::make_shared<Database>("TaxiRides", 134217728);
	Database::AddToInMemoryDatabaseList(database3);
	std::cout << "Loading trimmed-trips-part1.csv ..." << std::endl;
	csvDataImporter3.ImportTables(database3);

	CSVDataImporter csvDataImporter4(R"(../../data/trimmed-trips-part2.csv)");
	csvDataImporter4.SetTypes(types);
	csvDataImporter4.SetTableName(tableName);
	std::cout << "Loading trimmed-trips-part2.csv ..." << std::endl;
	csvDataImporter4.ImportTables(database3);

	Database::SaveAllToDisk();
	return 0;

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