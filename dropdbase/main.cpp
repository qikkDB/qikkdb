#include <cstdio>
#include <iostream>
#include <chrono>
#include <boost/log/core.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/trivial.hpp>
#include "QueryEngine/Context.h"
#include "Configuration.h"
#include "TCPServer.h"
#include "ClientPoolWorker.h"
#include "TCPClientHandler.h"
#include "ConsoleHandler.h"

int main(int argc, char **argv)
{
	Context::getInstance(); // Initialize CUDA context

    boost::log::add_file_log("../log/ColmnarDB.log");
    boost::log::add_console_log(std::cout);
	BOOST_LOG_TRIVIAL(info) << "Starting ColmnarDB...\n";
	Database::LoadDatabasesFromDisk();
	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	RegisterCtrlCHandler();
	tcpServer.Run();
	Database::SaveAllToDisk(Configuration::GetInstance().GetDatabaseDir().c_str());


    return 0;
}