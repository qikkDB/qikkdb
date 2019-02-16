#include <cstdio>
#include <iostream>
#include <chrono>

#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "QueryEngine/GPUCore/GPUFilter.cuh"
#include "QueryEngine/GPUCore/GPUDate.cuh"
#include "QueryEngine/GPUCore/cuda_ptr.h"

const int32_t TEST_EL_COUNT = 8;
int64_t testDateTimes[] = { 0, 1, 60, 3599, 3600, 5555, 86399, 86400, };
const int32_t correctYears[] = { 0, 0,  0,    0,    1,    1,    23,    24, };

int main(int argc, char **argv)
{
	std::unique_ptr<int32_t[]> resultHost = std::make_unique<int32_t[]>(TEST_EL_COUNT);
	cuda_ptr<int64_t> dtDevice(TEST_EL_COUNT);	// use our cuda smart pointer
	cuda_ptr<int32_t> resultDevice(TEST_EL_COUNT);

	GPUMemory::copyHostToDevice(dtDevice.get(), testDateTimes, TEST_EL_COUNT);
	GPUDate::extractCol<DateOperations::hour>(resultDevice.get(), dtDevice.get(), TEST_EL_COUNT);
	GPUMemory::copyDeviceToHost(resultHost.get(), resultDevice.get(), TEST_EL_COUNT);

	for (int i = 0; i < TEST_EL_COUNT; i++)
	{
		if (resultHost.get()[i] != correctYears[i])
			abort();
	}
	
	/*
	std::vector<std::string> tableNames = { "TableA" };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 5, false, tableNames, columnTypes);

	//GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();
	


    GpuSqlCustomParser parser(database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 <= 20;");
    parser.parse()->PrintDebugString();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed(end - start);

	std::cout << "Elapsed time: " << elapsed.count() << " s." << std::endl;
	*/
		
	return 0;
}
