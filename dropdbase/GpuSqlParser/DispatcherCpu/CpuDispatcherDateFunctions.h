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
	std::string colPointerNameMin;
	std::string colPointerNameMax;
	std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

	auto colValMin = allocatedPointers.at(colPointerNameMin);
	auto colValMax = allocatedPointers.at(colPointerNameMax);

	int32_t * resultMin = allocateRegister<int32_t>(reg + "_min", 1, std::get<2>(colValMin));
	int32_t * resultMax = allocateRegister<int32_t>(reg + "_max", 1, std::get<2>(colValMax));
	
	resultMin[0] = OP{}.operator()(reinterpret_cast<int64_t*>(std::get<0>(colValMin))[0]);
	resultMax[0] = OP{}.operator()(reinterpret_cast<int64_t*>(std::get<0>(colValMax))[0]);

	std::cout << "Where evaluation dateCol_min: " << colName << ", " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation dateCol_max: " << colName << ", " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}

template<typename OP>
int32_t CpuSqlDispatcher::dateExtractConst()
{
	auto cnst = arguments.read<int64_t>();
	auto reg = arguments.read<std::string>();

	int32_t * resultMin = allocateRegister<int32_t>(reg + "_min", 1, false);
	int32_t * resultMax = allocateRegister<int32_t>(reg + "_max", 1, false);

	resultMin[0] = OP{}.operator()(cnst);
	resultMax[0] = OP{}.operator()(cnst);

	std::cout << "Where evaluation dateConst_min: " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation dateConst_max: " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}