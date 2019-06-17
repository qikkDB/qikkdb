#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

template<typename OP, typename T>
int32_t CpuSqlDispatcher::arithmeticUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	loadCol<T>(colName);

	//TODO return type
	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int32_t * result = allocateRegister<int32_t>(reg, 1, std::get<2>(colVal));
	result[0] = OP{}.operator()(reinterpret_cast<T*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation arithmeticUnaryCol: " << colName << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::arithmeticUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	//TODO return type
	int32_t * result = allocateRegister<int32_t>(reg, 1, false);
	result[0] = OP{}.operator()(cnst);

	std::cout << "Where evaluation arithmeticUnaryConst: " << reg << ": " << result[0] << std::endl;

	return 0;
}