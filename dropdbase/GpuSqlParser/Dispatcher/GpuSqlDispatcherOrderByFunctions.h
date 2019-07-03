#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/OrderByType.h"
#include "../../VariantArray.h"

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

	orderByColumns.insert({ colName, order });
	
	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			std::cout << "Order by: " << colName << std::endl;
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(getAllocatedRegisterName(colName) + (groupByColumns.find(colName) != groupByColumns.end() ? "_keys" : ""));
			int32_t retSize = std::get<1>(column);

			if (orderByTable == nullptr)
			{
				orderByTable = std::make_unique<GPUOrderBy>(retSize);
				int32_t* orderByIndices = allocateRegister<int32_t>("$orderByIndices", retSize);
				usingOrderBy = true;
			}

			std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
			orderByTable->OrderByColumn(
				reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)),
				reinterpret_cast<T*>(std::get<0>(column)),
				retSize,
				order);
		}
		else
		{
			return 0;
		}
	}
	else
	{
		std::cout << "Order by: " << colName << std::endl;
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (orderByTable == nullptr)
		{
			orderByTable = std::make_unique<GPUOrderBy>(retSize);
			int32_t* orderByIndices = allocateRegister<int32_t>("$orderByIndices", retSize);
			usingOrderBy = true;
		}

		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
		orderByTable->OrderByColumn(
			reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)),
			reinterpret_cast<T*>(std::get<0>(column)),
			retSize,
			order);
	}	

	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByConst()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByReconstructCol()
{
	auto colName = arguments.read<std::string>();

	if (!usingGroupBy)
	{
		std::cout << "Reordering order by block: " << colName << std::endl;

		int32_t loadFlag = loadCol<T>(colName);
		if (loadFlag)
		{
			return loadFlag;
		}

		std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(getAllocatedRegisterName(colName));
		int32_t inSize = std::get<1>(col);

		std::unique_ptr<VariantArray<T>> outData = std::make_unique<VariantArray<T>>();
		outData->resize(inSize);

		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");

		GPUOrderBy::ReOrderByIdxInplace(reinterpret_cast<T*>(std::get<0>(col)), reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)), std::get<1>(col));
		int32_t outSize;
		GPUReconstruct::reconstructCol(outData->getData(), &outSize, reinterpret_cast<T*>(std::get<0>(col)), nullptr, inSize);
		reconstructedOrderByColumns[colName].push_back(std::move(outData));
	}
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByReconstructConst()
{
	return 0;
}


