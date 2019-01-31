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
	int vals[DATA_ELEMENT_COUNT] =     {   1,  -4,   5,   1,   1,  -1,   1,  -1 };
	//float vals[DATA_ELEMENT_COUNT] = { 1.5f, 1.2f, 0.01f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f };
	int32_t keys2[DATA_ELEMENT_COUNT] = { 'C', 'B', 'C', 'C', 'D', 'B', 'A', 'C' };
	int vals2[DATA_ELEMENT_COUNT] =     {   1,   1,   1,   1,  30,  -8,   1,   1 };
	//float vals2[DATA_ELEMENT_COUNT] = { 1.0f, 1.0f, 1.0f, 1.0f, 30.0f, 1.0f, 1.0f, 1.0f };
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
	GPUGroupBy<AggregationFunctions::avg, float, int32_t, int32_t> groupByInstance(HASH_TABLE_SIZE_MAX);
	groupByInstance.groupBy(p_keys, p_values, DATA_ELEMENT_COUNT);
	GPUMemory::copyHostToDevice(p_keys, keys2, DATA_ELEMENT_COUNT);
	GPUMemory::copyHostToDevice(p_values, vals2, DATA_ELEMENT_COUNT);
	groupByInstance.groupBy(p_keys, p_values, DATA_ELEMENT_COUNT);
	GPUMemory::free(p_keys);
	GPUMemory::free(p_values);
	// Get back results
	int32_t resultElementCount;
	int32_t *p_result_keys;
	float *p_result_values;
	GPUMemory::alloc(&p_result_keys, HASH_TABLE_SIZE_MAX);
	GPUMemory::alloc(&p_result_values, HASH_TABLE_SIZE_MAX);
	groupByInstance.getResults(p_result_keys, p_result_values, &resultElementCount);
	int32_t*  p_result_keys_cpu = new int32_t[resultElementCount];
	float*  p_result_values_cpu = new float[resultElementCount];
	GPUMemory::copyDeviceToHost(p_result_keys_cpu, p_result_keys, resultElementCount);
	GPUMemory::copyDeviceToHost(p_result_values_cpu, p_result_values, resultElementCount);
	GPUMemory::free(p_result_keys);
	GPUMemory::free(p_result_values);
	printf("Result data element count: %d\n", resultElementCount);
	for (int32_t i = 0; i < resultElementCount && i < DATA_ELEMENT_COUNT; i++)
	{
		printf("%c : %.2f\n", p_result_keys_cpu[i], p_result_values_cpu[i]);
	}

	delete[] p_result_keys_cpu;
	delete[] p_result_values_cpu;

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