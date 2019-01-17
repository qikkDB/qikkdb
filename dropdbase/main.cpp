

#include <cstdio>
#include <iostream>
#include <chrono>
#include <spdlog/spdlog.h>
#include <GpuSqlParser/GpuSqlCustomParser.h>

//#include "GpuSqlParser/MemoryStream.h"

int main(int argc, char **argv)
{
    spdlog::info("Application Starting");

    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Database> database(new Database());

    GpuSqlCustomParser parser(database, "SELECT COUNT(ageId) FROM TargetLoc100M WHERE latitude  > 48.163267512773274 AND latitude < 48.17608989851882 AND longitude > 17.19991468973717 AND longitude < 17.221200700479358 GROUP BY ageId;");
    parser.parse();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " s\n";


    return 0;
}