#pragma once
#include "../CpuSqlDispatcher.h"
#include <tuple>

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::filterColConst()
{
	auto colName = arguments.read<std::string>();
	U cnst = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	T min = getBlockMin(tableName, columnName);
	T max = getBlockMax(tableName, columnName);

	//TODO there is only implementation with min, but also should work with max
	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = OP{}.template operator() < T, U > (min, cnst);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::filterConstCol()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	T min = getBlockMin(tableName, columnName);
	T max = getBlockMax(tableName, columnName);

	//TODO there is only implementation with min, but also should work with max
	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = OP{}.template operator() < T, U > (cnst, min);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::filterColCol()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = 1;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();


	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = OP{}.template operator() < T, U > (constLeft, constRight);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalColConst()
{
	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalConstCol()
{
	return int32_t();
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalColCol()
{
	return int32_t();
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalConstConst()
{
	return int32_t();
}
