#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUDate.cuh"

template<typename OP>
int32_t CpuSqlDispatcher::dateExtractCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	loadCol<int64_t>(colName);

	//TODO ResultType
	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int32_t * result = allocateRegister<int32_t>(reg, 1, std::get<2>(colVal));
	result[0] = OP{}.operator()(reinterpret_cast<int64_t*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation dateCol" << (evaluateMin ? "_min" : "_max") << ": " << colName << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP>
int32_t CpuSqlDispatcher::dateExtractConst()
{
	auto cnst = arguments.read<int64_t>();
	auto reg = arguments.read<std::string>();

	int32_t * result = allocateRegister<int32_t>(reg, 1, false);
	result[0] = OP{}.operator()(cnst);

	std::cout << "Where evaluation dateConst" << (evaluateMin ? "_min" : "_max") << ": " << reg << ": " << result[0] << std::endl;

	return 0;
}