#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T>
int32_t GpuSqlDispatcher::arithmeticUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T

	typedef typename std::conditional < OP::isFloatRetType, float, T>::type ResultType;

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticUnaryCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(getAllocatedRegisterName(colName) + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(std::get<0>(column)), retSize);
			groupByColumns.insert({ reg, GpuSqlDispatcher::GetColumnType<ResultType>() });
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);
		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(std::get<0>(column)), retSize);
		}
	}
	freeColumnIfRegister<T>(colName);
	return 0;
}

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for constant case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T>
int32_t GpuSqlDispatcher::arithmeticUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename std::conditional < OP::isFloatRetType, float, T > ::type ResultType;

	std::cout << "ArithmeticUnaryConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		ResultType * result = allocateRegister<ResultType>(reg, retSize);
		GPUArithmeticUnary::cnst<OP, ResultType, T>(result, cnst, retSize);
	}

	return 0;
}