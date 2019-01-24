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

int main(int argc, char **argv)
{
	Context::getInstance(); // Initialize CUDA context

    boost::log::add_file_log("../log/ColmnarDB.log");
    boost::log::add_console_log(std::cout);

    std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, 1 << 24);

	auto start = std::chrono::high_resolution_clock::now();

    GpuSqlCustomParser parser(database, "SELECT colInteger FROM TableA WHERE colInteger = 200;");
    parser.parse();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " s\n";



    return 0;
}