#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUCast.cuh"

template<typename OUT, typename IN>
int32_t GpuSqlDispatcher::castNumericCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<IN>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "CastNumericCol: " << colName << " " << reg << std::endl;

	if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) != groupByColumns.end())
	{
		if (isOverallLastBlock)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			OUT *result = allocateRegister<OUT>(reg + "_keys", retSize);
			GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(std::get<0>(column)), retSize);
			groupByColumns.push_back({ reg, ::GetColumnType<OUT>() });
		}
	}
	else if (isOverallLastBlock || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			OUT *result = allocateRegister<OUT>(reg, retSize);
			GPUCast::CastNumericCol(result, reinterpret_cast<IN*>(std::get<0>(column)), retSize);
		}
	}

	freeColumnIfRegister<IN>(colName);
	return 0;
}

template<typename OUT, typename IN>
int32_t GpuSqlDispatcher::castNumericConst()
{
	IN cnst = arguments.read<IN>();
	auto reg = arguments.read<std::string>();

	std::cout << "CastNumericConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		OUT *result = allocateRegister<OUT>(reg, retSize);
		GPUCast::CastNumericConst(result, cnst, retSize);
	}
	return 0;
}