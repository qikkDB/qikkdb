#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

template<typename OP, typename T>
int32_t CpuSqlDispatcher::arithmeticUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();
	
	typedef typename std::conditional < OP::isFloatRetType, float, T>::type ResultType;

	loadCol<T>(colName);

	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	ResultType * result = allocateRegister<ResultType>(reg, 1, std::get<2>(colVal) || !OP::isMonotonous);
	result[0] = OP{}.template operator()< ResultType, T>(reinterpret_cast<T*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation arithmeticUnaryCol" << (evaluateMin ? "_min" : "_max") << ": " << reinterpret_cast<T*>(std::get<0>(colVal))[0] << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::arithmeticUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	typedef typename std::conditional < OP::isFloatRetType, float, T>::type ResultType;

	ResultType * result = allocateRegister<ResultType>(reg, 1, !OP::isMonotonous);
	result[0] = OP{}.template operator()< ResultType, T>(cnst);

	std::cout << "Where evaluation arithmeticUnaryConst" << (evaluateMin ? "_min" : "_max") << ": " <<  reg << ": " << result[0] << std::endl;

	return 0;
}