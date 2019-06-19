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

	std::string colPointerNameMin;
	std::string colPointerNameMax;
	std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

	auto colValMin = allocatedPointers.at(colPointerNameMin);
	auto colValMax = allocatedPointers.at(colPointerNameMax);

	ResultType * resultMin = allocateRegister<ResultType>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
	ResultType * resultMax = allocateRegister<ResultType>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);
	
	resultMin[0] = OP{}.template operator()< ResultType, T>(reinterpret_cast<T*>(std::get<0>(colValMin))[0]);
	resultMax[0] = OP{}.template operator()< ResultType, T>(reinterpret_cast<T*>(std::get<0>(colValMax))[0]);

	std::cout << "Where evaluation arithmeticUnaryCol_min: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0] << ", " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticUnaryCol_max: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0] << ", " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::arithmeticUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	typedef typename std::conditional < OP::isFloatRetType, float, T>::type ResultType;

	ResultType * resultMin = allocateRegister<ResultType>(reg + "_min", 1, !OP::isMonotonous);
	ResultType * resultMax = allocateRegister<ResultType>(reg + "_max", 1, !OP::isMonotonous);

	resultMin[0] = OP{}.template operator()< ResultType, T>(cnst);
	resultMax[0] = OP{}.template operator()< ResultType, T>(cnst);

	std::cout << "Where evaluation arithmeticUnaryConst_min: " <<  reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticUnaryConst_max: " <<  reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}