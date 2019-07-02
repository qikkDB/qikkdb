#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../../QueryEngine/OrderByType.h"

template<typename T>
int32_t GpuSqlDispatcher::orderByCol()
{
	auto colName = arguments.read<std::string>();
	OrderBy::Order order = static_cast<OrderBy::Order>(arguments.read<int32_t>());

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (orderByTable)
	{
		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
		orderByTable->OrderByColumn(
			reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)),
			reinterpret_cast<T*>(std::get<0>(column)), 
			retSize, 
			order);
	}
	else
	{
		orderByTable = std::make_unique<GPUOrderBy>(retSize);
		allocateRegister<int32_t>("$orderByIndices", retSize);
		usingOrderBy = true;
	}

	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByConst()
{
	return 0;
}