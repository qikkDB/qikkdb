#include <cstdio>
#include <iostream>
#include <chrono>
#include <spdlog/spdlog.h>
#include <GpuSqlParser/GpuSqlCustomParser.h>
#include "GpuSqlParser/MemoryStream.h"

int main(int argc, char **argv)
{
    spdlog::info("Application Starting");

    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Database> database(new Database());

    GpuSqlCustomParser parser(database, "SELECT abc.b FROM abc WHERE ((abc.a = 1) AND (abc.b = 2)) OR (abc.c = 3);");
    parser.parse();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    MemoryStream memoryStream;

    memoryStream.insert<int>(5);
    memoryStream.insert<float>(5.5f);
    memoryStream.insert<const std::string&>("Hello guys");

    std::cout << memoryStream.read<int>() << std::endl;
    std::cout << memoryStream.read<float>() << std::endl;
    std::cout << memoryStream.read<std::string>() << std::endl;


    return 0;
}
