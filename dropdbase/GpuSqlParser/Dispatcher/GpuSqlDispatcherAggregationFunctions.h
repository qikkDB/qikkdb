#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUGroupBy.cuh"
#include "../../QueryEngine/GPUCore/GPUGroupByString.cuh"
#include "../../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"

/// Implementation of generic aggregation operation based on functor OP
/// Used when GROUP BY Clause is not present
/// Loads data on demand
/// COUNT operation is handled more efficiently
/// If WHERE clause is present filtering is done before agreggation
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename OUT, typename IN>
int32_t GpuSqlDispatcher::aggregationCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<IN>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(colName);
	int32_t reconstructOutSize;

	IN* reconstructOutReg = nullptr;
	if (std::is_same<OP, AggregationFunctions::count>::value)
	{
		// If mask is present - count suitable rows
		if (filter_)
		{
			int32_t *indexes = nullptr;
			GPUReconstruct::GenerateIndexesKeep(&indexes, &reconstructOutSize, reinterpret_cast<int8_t*>(filter_), std::get<1>(column));
			if (indexes)
			{
				GPUMemory::free(indexes);
			}
		}
		// If mask is nullptr - count full rows
		else {
			reconstructOutSize = std::get<1>(column);
		}
	}
	else
	{
		GPUReconstruct::reconstructColKeep<IN>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<IN*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));
	}

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
		// TODO: if (not COUNT operation and std::get<1>(column) == 0), set result to NaN
		OUT * result = allocateRegister<OUT>(reg, 1);
		GPUAggregation::col<OP, OUT, IN>(result, reinterpret_cast<IN*>(std::get<0>(column)), std::get<1>(column));
	}
	freeColumnIfRegister<IN>(colName);
	filter_ = 0;
	return 0;
}

template<typename OP, typename OUT, typename IN>
int32_t GpuSqlDispatcher::aggregationConst()
{
	std::cout << "AggConst" << std::endl;
	return 0;
}

template<typename OP, typename O, typename K, typename V>
class GpuSqlDispatcher::GroupByHelper
{
public:
	static void ProcessBlock(const std::string& groupByColumnName, const std::tuple<uintptr_t, int32_t, bool>& valueColumn, GpuSqlDispatcher& dispatcher)
	{
		std::tuple<uintptr_t, int32_t, bool> groupByColumn = dispatcher.allocatedPointers.at(groupByColumnName);

		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(valueColumn));

		reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->groupBy(reinterpret_cast<K*>(std::get<0>(groupByColumn)), reinterpret_cast<V*>(std::get<0>(valueColumn)), dataSize);
	}

	static void GetResults(const std::string& groupByColumnName, const std::string& reg, GpuSqlDispatcher& dispatcher)
	{
		int32_t outSize;
		K* outKeys = nullptr;
		O* outValues = nullptr;
		reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->getResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables);
		dispatcher.allocatedPointers.insert({ groupByColumnName + "_keys",std::make_tuple(reinterpret_cast<uintptr_t>(outKeys), outSize, true) });
		dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<uintptr_t>(outValues), outSize, true) });
	}
};

template<typename OP, typename O, typename V>
class GpuSqlDispatcher::GroupByHelper<OP, O, std::string, V>
{
public:
	static void ProcessBlock(const std::string& groupByColumnName, std::tuple<uintptr_t, int32_t, bool>& valueColumn, GpuSqlDispatcher& dispatcher)
	{
		auto groupByColumn = dispatcher.findStringColumn(groupByColumnName);

		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(valueColumn));

		reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->groupBy(std::get<0>(groupByColumn), reinterpret_cast<V*>(std::get<0>(valueColumn)), dataSize);
	}
	
	static void GetResults(const std::string& groupByColumnName, const std::string& reg, GpuSqlDispatcher& dispatcher)
	{
		int32_t outSize;
		GPUMemory::GPUString outKeys;
		V* outValues = nullptr;
		reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->getResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables);
		dispatcher.fillStringRegister(outKeys, groupByColumnName + "_keys", outSize, true);
		dispatcher.allocatedPointers.insert({ reg,std::make_tuple(reinterpret_cast<uintptr_t>(outValues), outSize, true) });
	}
};

/// Implementation of generic aggregation operation based on functor OP
/// Used when GROUP BY Clause is present
/// Loads data on demand
/// If WHERE clause is present filtering is done before agreggation
/// For each block it updates group by hash table
/// To handle multi-gpu functionality - each dipatcher instance signals when it processes its last block
/// The dispatcher instance handling the overall last block waits for all other dispatcher instances to finish their last blocks
/// and saves the result of group by
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename R, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationGroupBy()
{
	auto colTableName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colTableName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggGroupBy: " << colTableName << " " << reg << ", thread: " << dispatcherThreadId << std::endl;


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

	//TODO void param
	if (groupByTables[dispatcherThreadId] == nullptr)
	{
		groupByTables[dispatcherThreadId] = std::make_unique<GPUGroupBy<OP, R, U, T>>(Configuration::GetInstance().GetGroupByBuckets());
	}

	std::string groupByColumnName = *(groupByColumns.begin());
	
	GpuSqlDispatcher::GroupByHelper<OP, R, U, T>::ProcessBlock(groupByColumnName, column, *this);
	
	// If last block was processed, reconstruct group by table
	if (isLastBlockOfDevice)
	{
		if (isOverallLastBlock)
		{
			// Wait until all threads finished work
			std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
			GpuSqlDispatcher::groupByCV_.wait(lock, [] { return GpuSqlDispatcher::IsGroupByDone(); });

			std::cout << "Reconstructing group by in thread: " << dispatcherThreadId << std::endl;
			
			GpuSqlDispatcher::GroupByHelper<OP, R, U, T>::GetResults(groupByColumnName, reg, *this);
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
	
	freeColumnIfRegister<U>(colTableName);
	return 0;
}

/// This executes first (dor each block) when GROUP BY clause is used
/// It loads the group by column (if it is firt encountered reference to the column)
/// and filters it according to WHERE clause
/// <returns name="statusCode">Finish status code of the operation</returns>
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
