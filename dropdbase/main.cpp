/*
#include <cstdio>
#include <iostream>
#include <chrono>
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include "GpuSqlParser/MemoryStream.h"
#include <boost/log/core.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>

#include "DatabaseGenerator.h"*/
#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "QueryEngine/GPUCore/GPUGroupBy.cuh"



int main(int argc, char **argv)
{
	Context::getInstance(); // Initialize CUDA context

	const int32_t HASH_TABLE_SIZE_MAX = 65536;

	const int32_t DATA_ELEMENT_COUNT = 8;
	int32_t keys[DATA_ELEMENT_COUNT] = { 'A', 'B', 'A', 'C', 'D', 'B', 'A', 'C' };
	int32_t vals[DATA_ELEMENT_COUNT] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	/*
	 * A : 3
	 * B : 2
	 * C : 2
	 * D : 1
	 */

	int32_t *p_keys;
	int32_t *p_values;

	GPUMemory::alloc(&p_keys, DATA_ELEMENT_COUNT);
	GPUMemory::alloc(&p_values, DATA_ELEMENT_COUNT);

	GPUMemory::copyHostToDevice(p_keys, keys, DATA_ELEMENT_COUNT);
	GPUMemory::copyHostToDevice(p_values, vals, DATA_ELEMENT_COUNT);

	GPUGroupBy<AggregationFunctions::sum, int32_t, int32_t> groupByInstance(HASH_TABLE_SIZE_MAX);
	groupByInstance.groupBy(p_keys, p_values, DATA_ELEMENT_COUNT);

	int32_t resultElementCount = groupByInstance.getResultElementCount();
	int32_t *p_result_keys = new int32_t[resultElementCount];
	int32_t *p_result_values = new int32_t[resultElementCount];

	groupByInstance.getResults(p_result_keys, p_result_values);

	printf("Result data element count: %d\n", resultElementCount);

	for (int32_t i = 0; i < resultElementCount; i++)
	{
		printf("%c : %d\n", p_result_keys[i], p_result_values[i]);
	}

	GPUMemory::free(p_keys);
	GPUMemory::free(p_values);

	delete[] p_result_keys;
	delete[] p_result_values;
	/*
	boost::log::add_file_log("../log/ColmnarDB.log");
	boost::log::add_console_log(std::cout);

	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 24);
	GPUMemory::hostPin(dynamic_cast<BlockBase<int32_t>&>(*dynamic_cast<ColumnBase<int32_t>&>(*(database->GetTables().at("TableA").GetColumns().at("colInteger"))).GetBlocksList()[0]).GetData().data(), 1 << 24);
	auto start = std::chrono::high_resolution_clock::now();

	GpuSqlCustomParser parser(database, "SELECT colInteger FROM TableA WHERE (colInteger + 2) * 2 <= 20;");
	parser.parse();

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed(end - start);

	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	*/
	return 0;
}