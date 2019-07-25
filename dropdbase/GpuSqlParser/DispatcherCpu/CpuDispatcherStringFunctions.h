#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/StringOperations.h"

template<typename OP>
int32_t CpuSqlDispatcher::stringUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	if (loadCol<std::string>(colName))
	{
		return 1;
	}

	std::string colPointerNameMin;
	std::string colPointerNameMax;
	std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

	auto colValMin = allocatedPointers.at(colPointerNameMin);
	auto colValMax = allocatedPointers.at(colPointerNameMax);

	std::string resultStringMin = OP{}(reinterpret_cast<char*>(std::get<0>(colValMin)), std::get<1>(colValMin));
	std::string resultStringMax = OP{}(reinterpret_cast<char*>(std::get<0>(colValMax)), std::get<1>(colValMax));

	char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1, std::get<2>(colValMin));
	char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1, std::get<2>(colValMax));

	std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
	resultMin[resultStringMin.size()] = '\0';
	std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
	resultMax[resultStringMax.size()] = '\0';

	return 0;
}


template<typename OP>
int32_t CpuSqlDispatcher::stringUnaryConst()
{
	auto cnst = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::string resultStringMin = OP{}(cnst.c_str(), cnst.size());
	std::string resultStringMax = OP{}(cnst.c_str(), cnst.size());

	char* resultMin = allocateRegister<char>(reg + "_min", resultStringMin.size() + 1, false);
	char* resultMax = allocateRegister<char>(reg + "_max", resultStringMax.size() + 1, false);

	std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
	resultMin[resultStringMin.size()] = '\0';
	std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
	resultMax[resultStringMax.size()] = '\0';

	return 0;
}

/*
template<typename OP>
int32_t CpuSqlDispatcher::stringUnaryNumericCol()
{

}

template<typename OP>
int32_t CpuSqlDispatcher::stringUnaryNumericConst()
{

}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringBinaryNumericColCol()
{

}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericColConst()
{

}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericConstCol()
{

}

template<typename OP, typename T>
int32_t CpuSqlDispatcher::stringBinaryNumericConstConst()
{

}

template<typename OP>
int32_t CpuSqlDispatcher::stringBinaryColCol()
{

}

template<typename OP>
int32_t CpuSqlDispatcher::stringBinaryColConst()
{

}

template<typename OP>
int32_t CpuSqlDispatcher::stringBinaryConstCol()
{

}

template<typename OP>
int32_t CpuSqlDispatcher::stringBinaryConstConst()
{

}
*/