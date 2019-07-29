#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/OrderByType.h"
#include "../../QueryEngine/GPUCore/cuda_ptr.h"
#include "../../VariantArray.h"

template<typename T>
int32_t GpuSqlDispatcher::orderByCol()
{
	auto colName = arguments.read<std::string>();
	OrderBy::Order order = static_cast<OrderBy::Order>(arguments.read<int32_t>());
	int32_t columnIndex = arguments.read<int32_t>();

	orderByColumns.insert({ columnIndex, {colName, order} });

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}
	
	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			std::cout << "Order by: " << colName << std::endl;
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName +
					(std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) != groupByColumns.end() ?
					 "_keys" : ""));
			int32_t inSize = std::get<1>(column);

			if (orderByTable == nullptr)
			{
				orderByTable = std::make_unique<GPUOrderBy>(inSize);
				int32_t* orderByIndices = allocateRegister<int32_t>("$orderByIndices", inSize);
				usingOrderBy = true;
			}

			std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
			dynamic_cast<GPUOrderBy*>(orderByTable.get())->OrderByColumn(
				reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)),
				reinterpret_cast<T*>(std::get<0>(column)),
				inSize,
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
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t inSize = std::get<1>(column);

		if (orderByTable == nullptr)
		{
			orderByTable = std::make_unique<GPUOrderBy>(inSize);
			int32_t* orderByIndices = allocateRegister<int32_t>("$orderByIndices", inSize);
			usingOrderBy = true;
		}

		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
		dynamic_cast<GPUOrderBy*>(orderByTable.get())->OrderByColumn(
			reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)),
			reinterpret_cast<T*>(std::get<0>(column)),
			inSize,
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
int32_t GpuSqlDispatcher::orderByReconstructOrderCol()
{
	auto colName = arguments.read<std::string>();

	if (!usingGroupBy)
	{
		std::cout << "Reordering order by column: " << colName << std::endl;

		int32_t loadFlag = loadCol<T>(colName);
		if (loadFlag)
		{
			return loadFlag;
		}

		std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(colName);
		int32_t inSize = std::get<1>(col);

		std::unique_ptr<VariantArray<T>> outData = std::make_unique<VariantArray<T>>(inSize);

		cuda_ptr<T> reorderedColumn(inSize);

		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
		GPUOrderBy::ReOrderByIdx(reorderedColumn.get(), reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)), reinterpret_cast<T*>(std::get<0>(col)), std::get<1>(col));
		
		int32_t outSize;
		GPUReconstruct::reconstructCol(outData->getData(), &outSize, reorderedColumn.get(), reinterpret_cast<int8_t*>(filter_) , inSize);
		outData->resize(outSize);

		orderByBlocks[dispatcherThreadId].reconstructedOrderByOrderColumnBlocks[colName].push_back(std::move(outData));
	}
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByReconstructOrderConst()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByReconstructRetCol()
{
	auto colName = arguments.read<std::string>();

	if (!usingGroupBy)
	{
		std::cout << "Recostructing order by return column: " << colName << std::endl;

		int32_t loadFlag = loadCol<T>(colName);
		if (loadFlag)
		{
			return loadFlag;
		}

		std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(colName);
		int32_t inSize = std::get<1>(col);

		std::unique_ptr<VariantArray<T>> outData = std::make_unique<VariantArray<T>>(inSize);

		cuda_ptr<T> reorderedColumn(inSize);

		std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
		GPUOrderBy::ReOrderByIdx(reorderedColumn.get(), reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)), reinterpret_cast<T*>(std::get<0>(col)), std::get<1>(col));

		int32_t outSize;
		//GPUReconstruct::reconstructCol(outData->getData(), &outSize, reinterpret_cast<T*>(std::get<0>(col)), reinterpret_cast<int8_t*>(filter_), inSize);
		GPUReconstruct::reconstructCol(outData->getData(), &outSize, reorderedColumn.get(), reinterpret_cast<int8_t*>(filter_) , inSize);
		outData->resize(outSize);

		orderByBlocks[dispatcherThreadId].reconstructedOrderByRetColumnBlocks[colName].push_back(std::move(outData));
	}
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::orderByReconstructRetConst()
{
	return 0;
}
