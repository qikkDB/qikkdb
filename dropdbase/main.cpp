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
#include <boost/log/utility/setup/from_stream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/formatter_parser.hpp>
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
/// <param name="argv">program arguments (for CSV importing): [csv-path [new-db-name]],
/// 		for taxi rides import use switch -t</param>
/// <returns>Exit code (0 - OK)</returns>
int main(int argc, char **argv)
{
	// Logger setup
	boost::log::add_common_attributes();
	boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");
	std::ifstream logConfigFile("../configuration/log_config");
	if (logConfigFile.fail())
	{
		logConfigFile = std::ifstream("../configuration/log_config.default");
	}
	boost::log::init_from_stream(logConfigFile);
	
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

	if (argc > 1)	// Importing CSV
	{
		if (strcmp(argv[1], "-t") == 0)
		{
			BOOST_LOG_TRIVIAL(info) << "Importing databases for testing has started (3 databases will be loaded)...";

			CSVDataImporter csvDataImporter1(R"(../../data/GeoPoint.csv)");
			std::shared_ptr<Database> database1 = std::make_shared<Database>("GeoTest", 131072);
			Database::AddToInMemoryDatabaseList(database1);
			BOOST_LOG_TRIVIAL(info) << "Loading GeoPoint.csv ...";
			csvDataImporter1.ImportTables(database1);

			CSVDataImporter csvDataImporter2(R"(../../data/TargetLoc1B.csv)");
			std::shared_ptr<Database> database2 = std::make_shared<Database>("TargetLocator", 134217728);
			Database::AddToInMemoryDatabaseList(database2);
			BOOST_LOG_TRIVIAL(info) << "Loading TargetLoc1B.csv ...";
			csvDataImporter2.ImportTables(database2);

			CSVDataImporter csvDataImporter3(R"(../../data/latest-trips-part1.csv)");
			const std::vector<DataType> types{
				COLUMN_STRING,
				COLUMN_LONG,
				COLUMN_LONG,
				COLUMN_INT,
				COLUMN_DOUBLE,
				COLUMN_DOUBLE,
				COLUMN_INT
			};
			const std::string tableName = "trips";
			csvDataImporter3.SetTypes(types);
			csvDataImporter3.SetTableName(tableName);
			std::shared_ptr<Database> database3 = std::make_shared<Database>("TaxiRides", 134217728);
			Database::AddToInMemoryDatabaseList(database3);
			BOOST_LOG_TRIVIAL(info) << "Loading latest-trips-part1.csv ...";
			csvDataImporter3.ImportTables(database3);

			CSVDataImporter csvDataImporter4(R"(../../data/latest-trips-part2.csv)");
			csvDataImporter4.SetTypes(types);
			csvDataImporter4.SetTableName(tableName);
			BOOST_LOG_TRIVIAL(info) << "Loading latest-trips-part2.csv ...";
			csvDataImporter4.ImportTables(database3);
		}
		else
		{
			if (strcmp(argv[1], "-a") == 0)
			{
				BOOST_LOG_TRIVIAL(info) << "Importing all databases has started (6 databases will be loaded)...";

				CSVDataImporter csvDataImporter1(R"(../../data/GeoPoint.csv)");
				std::shared_ptr<Database> database1 = std::make_shared<Database>("GeoTest", 131072);
				Database::AddToInMemoryDatabaseList(database1);
				BOOST_LOG_TRIVIAL(info) << "Loading GeoPoint.csv ...";
				csvDataImporter1.ImportTables(database1);

				CSVDataImporter csvDataImporter2(R"(../../data/TargetLoc1B.csv)");
				std::shared_ptr<Database> database2 = std::make_shared<Database>("TargetLocator", 134217728);
				Database::AddToInMemoryDatabaseList(database2);
				BOOST_LOG_TRIVIAL(info) << "Loading TargetLoc1B.csv ...";
				csvDataImporter2.ImportTables(database2);

				CSVDataImporter csvDataImporter3(R"(../../data/latest-trips-part1.csv)");
				const std::vector<DataType> types{
					COLUMN_STRING,
					COLUMN_LONG,
					COLUMN_LONG,
					COLUMN_INT,
					COLUMN_DOUBLE,
					COLUMN_DOUBLE,
					COLUMN_INT
				};
				const std::string tableName2 = "trips";
				csvDataImporter3.SetTypes(types);
				csvDataImporter3.SetTableName(tableName2);
				std::shared_ptr<Database> database3 = std::make_shared<Database>("TaxiRides", 134217728);
				Database::AddToInMemoryDatabaseList(database3);
				BOOST_LOG_TRIVIAL(info) << "Loading latest-trips-part1.csv ...";
				csvDataImporter3.ImportTables(database3);

				CSVDataImporter csvDataImporter4(R"(../../data/latest-trips-part2.csv)");
				csvDataImporter4.SetTypes(types);
				csvDataImporter4.SetTableName(tableName2);
				BOOST_LOG_TRIVIAL(info) << "Loading latest-trips-part2.csv ...";
				csvDataImporter4.ImportTables(database3);

				CSVDataImporter csvDataImporter5(R"(../../data/Target.csv)");
				std::shared_ptr<Database> database5 = std::make_shared<Database>("Target", 2097152);
				Database::AddToInMemoryDatabaseList(database5);
				BOOST_LOG_TRIVIAL(info) << "Loading Target.csv ...";
				csvDataImporter5.ImportTables(database5);

				CSVDataImporter csvDataImporter6(R"(../../data/TargetTraffic.csv)");
				std::shared_ptr<Database> database6 = std::make_shared<Database>("TargetTraffic", 16777216);
				Database::AddToInMemoryDatabaseList(database6);
				BOOST_LOG_TRIVIAL(info) << "Loading TargetTraffic.csv ...";
				csvDataImporter6.ImportTables(database6);

				CSVDataImporter csvDataImporter7(R"(../../data/D_Cell.csv)");
				std::shared_ptr<Database> database7 = std::make_shared<Database>("D_Cell", 8192);
				Database::AddToInMemoryDatabaseList(database7);
				BOOST_LOG_TRIVIAL(info) << "Loading D_Cell.csv ...";
				csvDataImporter7.ImportTables(database7);
			}
			else
			{
				// Import CSV file if entered as program argument
				CSVDataImporter csvDataImporter(argv[1]);
				////CSVDataImporter csvDataImporter(R"(D:\DataGenerator\output\TargetLoc1B.csv)");
				std::shared_ptr<Database> database = std::make_shared<Database>(argc > 2 ? argv[2] : "TestDb", argc > 3 ? std::stoll(argv[3]) : 1048576);
				Database::AddToInMemoryDatabaseList(database);
				BOOST_LOG_TRIVIAL(info) << "Loading CSV from \"" << argv[1] << "\"";
				csvDataImporter.ImportTables(database);
			}
		}
	}
	else	// TCP server
	{
		BOOST_LOG_TRIVIAL(info) << "Loading databases...";
		Database::LoadDatabasesFromDisk();
		BOOST_LOG_TRIVIAL(info) << "All databases loaded.";

		TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
		RegisterCtrlCHandler(&tcpServer);
		tcpServer.Run();
	}

	Database::SaveAllToDisk();
	BOOST_LOG_TRIVIAL(info) << "Exiting cleanly...";

	/*
	CSVDataImporter csvDataImporter(R"(D:\testing-data\TargetLoc100M.csv)");
	std::shared_ptr<Database> database = std::make_shared<Database>("TestDb", 100000000);
	Database::AddToInMemoryDatabaseList(database);
	BOOST_LOG_TRIVIAL(info) << "Loading TargetLoc.csv ...";
	csvDataImporter.ImportTables(database);
	*/

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
	BOOST_LOG_TRIVIAL(info) << "TellStoryDB exited.";
	return 0;
}
