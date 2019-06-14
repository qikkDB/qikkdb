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

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	T colVal = evaluateMin ? getBlockMin<T>(tableName, columnName) : getBlockMax<T>(tableName, columnName);
	*mask = OP{}.template operator() < T, U > (colVal, cnst);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::filterConstCol()
{
	T cnst = arguments.read<T>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	U colVal = evaluateMin ? getBlockMin<U>(tableName, columnName) : getBlockMax<U>(tableName, columnName);
	*mask = OP{}.template operator() < T, U > (cnst, colVal);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::filterColCol()
{
	auto colNameLeft = arguments.read<std::string>();
	auto colNameRight = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = 1;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterConstConst()
{
	T constLeft = arguments.read<T>();
	U constRight = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = OP{}.template operator() < T, U > (constLeft, constRight);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalColConst()
{
	auto colName = arguments.read<std::string>();
	U cnst = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	T colVal = evaluateMin ? getBlockMin<T>(tableName, columnName) : getBlockMax<T>(tableName, columnName);
	*mask = OP{}.template operator() < T, U > (colVal, cnst);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalConstCol()
{
	T cnst = arguments.read<T>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	U colVal = evaluateMin ? getBlockMin<U>(tableName, columnName) : getBlockMax<U>(tableName, columnName);
	*mask = OP{}.template operator() < T, U > (cnst, colVal);

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalColCol()
{
	auto colNameLeft = arguments.read<std::string>();
	auto colNameRight = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = 1;

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::logicalConstConst()
{
	T constLeft = arguments.read<T>();
	U constRight = arguments.read<U>();
	auto reg = arguments.read<std::string>();


	int8_t * mask = allocateRegister<int8_t>(reg, 1);
	*mask = OP{}.template operator() < T, U > (constLeft, constRight);

	return 0;
}
