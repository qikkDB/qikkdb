#pragma once

#include "../ParserExceptions.h"
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T

	typedef typename OP::returnType ResultType;

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}
	
	std::cout << "StringUnaryCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto column = findStringColumn(colName);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringUnary::Col<OP>(result, std::get<0>(column), std::get<1>(column));
			fillStringRegister(result, reg, std::get<1>(column));
		}
	}
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename OP::returnType ResultType;

	std::cout << "StringUnaryConst: " << reg << std::endl;

	GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString result;
		GPUStringUnary::Const<OP>(result, gpuString, retSize);
		fillStringRegister(result, reg, retSize);
	}
	return 0;
}