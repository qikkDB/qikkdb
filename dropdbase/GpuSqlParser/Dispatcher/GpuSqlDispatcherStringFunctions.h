#pragma once

#include "../ParserExceptions.h"
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"

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

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::stringBinaryColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename OP::returnType ResultType;

	std::cout << "StringBinaryColCol: " << reg << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::stringBinaryColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename OP::returnType ResultType;

	std::cout << "StringBinaryColConst: " << reg << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::stringBinaryConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename OP::returnType ResultType;

	std::cout << "StringBinaryConstCol: " << reg << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::stringBinaryConstConst()
{
	U cnstRight = arguments.read<U>();
	T cnstLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename OP::returnType ResultType;

	std::cout << "StringBinaryConstConst: " << reg << std::endl;

	return 0;
}