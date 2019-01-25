#include <cstdio>
#include <iostream>
#include <chrono>
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include "GpuSqlParser/MemoryStream.h"
#include <boost/log/core.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include "DatabaseGenerator.h"
#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "Configuration.h"
#include "TCPServer.h"
#include "ClientPoolWorker.h"
#include "TCPClientHandler.h"

int main(int argc, char **argv)
{
	Context::getInstance(); // Initialize CUDA context

    boost::log::add_file_log("../log/ColmnarDB.log");
    boost::log::add_console_log(std::cout);

	Database::LoadDatabasesFromDisk();
	TCPServer<TCPClientHandler, ClientPoolWorker> tcpServer(Configuration::GetInstance().GetListenIP().c_str(), Configuration::GetInstance().GetListenPort());
	Database::SaveAllToDisk(Configuration::GetInstance().GetDatabaseDir().c_str());


    return 0;
}