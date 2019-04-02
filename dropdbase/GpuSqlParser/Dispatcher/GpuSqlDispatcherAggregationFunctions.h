#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUGroupBy.cuh"
#include "../../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"

template<typename OP, typename R, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationColCol()
{
	auto colTableName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colTableName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggColCol: " << colTableName << " " << reg << ", thread: " << dispatcherThreadId << std::endl;


	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(colTableName);
	int32_t reconstructOutSize;

	if (!usingGroupBy || colTableName != *(groupByColumns.begin()))
	{
		T* reconstructOutReg;
		GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

		if (std::get<2>(column))
		{
			GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
		}
		else
		{
			std::get<2>(column) = true;
		}
		std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
		std::get<1>(column) = reconstructOutSize;
	}
	const size_t endOfPolyIdx = colTableName.find(".");
	const std::string table = colTableName.substr(0, endOfPolyIdx);
	const std::string columnName = colTableName.substr(endOfPolyIdx + 1);

	if (usingGroupBy)
	{
		//TODO void param
		if (groupByTables[dispatcherThreadId] == nullptr)
		{
			groupByTables[dispatcherThreadId] = std::make_unique<GPUGroupBy<OP, R, U, T>>(Configuration::GetInstance().GetGroupByBuckets());
		}

		std::string groupByColumnName = *(groupByColumns.begin());
		std::tuple<uintptr_t, int32_t, bool> groupByColumn = allocatedPointers.at(groupByColumnName);



		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(column));

		reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(groupByTables[dispatcherThreadId].get())->groupBy(reinterpret_cast<U*>(std::get<0>(groupByColumn)), reinterpret_cast<T*>(std::get<0>(column)), dataSize);

		// If last block was processed, reconstruct group by table
		if (isLastBlockOfDevice)
		{
			if (isOverallLastBlock)
			{
				// Wait until all threads finished work
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::groupByCV_.wait(lock, [] { return GpuSqlDispatcher::IsGroupByDone(); });

				std::cout << "Reconstructing group by in thread: " << dispatcherThreadId << std::endl;
				int32_t outSize;
				U* outKeys;
				R* outValues;
				reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(groupByTables[dispatcherThreadId].get())->getResults(&outKeys, &outValues, &outSize, groupByTables);
				allocatedPointers.insert({ groupByColumnName + "_keys",std::make_tuple(reinterpret_cast<uintptr_t>(outKeys), outSize, true) });
				allocatedPointers.insert({ reg,std::make_tuple(reinterpret_cast<uintptr_t>(outValues), outSize, true) });
			}
			else
			{
				std::cout << "Group by all blocks done in thread: " << dispatcherThreadId << std::endl;
				// Increment counter and notify threads
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::IncGroupByDoneCounter();
				GpuSqlDispatcher::groupByCV_.notify_all();
			}
		}
	}
	else
	{
		if (!isRegisterAllocated(reg))
		{
			T * result = allocateRegister<T>(reg, 1);
			GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
		}
	}
	freeColumnIfRegister<U>(colTableName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationColConst()
{
	std::cout << "AggColConst" << std::endl;
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationConstCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggConstCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(colName);
	int32_t reconstructOutSize;

	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

	if (std::get<2>(column))
	{
		GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	}
	else
	{
		std::get<2>(column) = true;
	}

	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	if (!isRegisterAllocated(reg))
	{
		T * result = allocateRegister<T>(reg, 1);
		GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	}
	freeColumnIfRegister<T>(colName);
	filter_ = 0;
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationConstConst()
{
	std::cout << "AggConstConst" << std::endl;
	return 0;
}


template<typename T>
int32_t GpuSqlDispatcher::groupByCol()
{
	std::string columnName = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(columnName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "GroupBy: " << columnName << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(columnName);

	int32_t reconstructOutSize;
	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

	if (std::get<2>(column))
	{
		GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	}
	else
	{
		std::get<2>(column) = true;
	}
	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	if (groupByColumns.find(columnName) == groupByColumns.end())
	{
		groupByColumns.insert(columnName);
	}
	usingGroupBy = true;
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::groupByConst()
{
	return 0;
}