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
#include "Version.h"

/// Startup function, called automatically.
/// <param name="argc">program argument count</param>
/// <param name="argv">program arguments (for CSV importing): [csv-path [new-db-name]],
/// 		for taxi rides import use switch -t</param>
/// <returns>Exit code (0 - OK)</returns>
int main(int argc, char** argv)
{
    // Logger setup
    boost::log::add_common_attributes();
    boost::log::register_simple_formatter_factory<boost::log::trivial::severity_level, char>(
        "Severity");
    std::ifstream logConfigFile("../configuration/log_config");
    if (logConfigFile.fail())
    {
        logConfigFile = std::ifstream("../configuration/log_config.default");
    }
    if (logConfigFile.fail())
    {
        BOOST_LOG_TRIVIAL(error)
            << "ERROR: Failed to load log configuration in \"../configuration/log_config.default\"";
    }
    else
    {
        boost::log::init_from_stream(logConfigFile);
    }

    BOOST_LOG_TRIVIAL(info) << "qikkDB " << CMAKE_BUILD_TYPE << " " << GIT_VERSION << GIT_BRANCH;
    Context::getInstance(); // Initialize CUDA context

    if (argc > 1) // Importing CSV
    {
        // Import CSV file if entered as program argument
        CSVDataImporter csvDataImporter(argv[1]);
        std::shared_ptr<Database> database =
            std::make_shared<Database>(argc > 2 ? argv[2] : "TestDb", argc > 3 ? std::stoll(argv[3]) : 1048576);
        Database::AddToInMemoryDatabaseList(database);
        BOOST_LOG_TRIVIAL(info) << "Loading CSV from \"" << argv[1] << "\"";
        csvDataImporter.ImportTables(database);
    }
    else // TCP server
    {
        BOOST_LOG_TRIVIAL(info)
            << "Loading databases from: " << Configuration::GetInstance().GetDatabaseDir();
        Database::LoadDatabasesFromDisk();
        BOOST_LOG_TRIVIAL(info) << "All databases from "
                                << Configuration::GetInstance().GetDatabaseDir() << " have been loaded.";

        TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(
            Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
        RegisterCtrlCHandler(&tcpServer);
        tcpServer.Run();
    }

    Database::SaveModifiedToDisk();
    BOOST_LOG_TRIVIAL(info) << "qikkDB exiting cleanly...";

    for (auto& db : Database::GetDatabaseNames())
    {
        Database::RemoveFromInMemoryDatabaseList(db.c_str());
    }

    BOOST_LOG_TRIVIAL(info) << "qikkDB exited.";
    boost::log::core::get()->remove_all_sinks();
    return 0;
}
