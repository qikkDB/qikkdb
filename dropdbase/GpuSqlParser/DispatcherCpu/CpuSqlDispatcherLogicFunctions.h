#pragma once

#include "../CpuSqlDispatcher.h"
#include <tuple>

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterColConst()
{
	auto colName = arguments.read<std::string>();
	U cnst = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	loadCol<T>(colName);

	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1, std::get<2>(colVal));

	mask[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colVal))[0], cnst);

	std::cout << "Where evaluation filterColConst: " << colName << ", " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterConstCol()
{
	T cnst = arguments.read<T>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	loadCol<U>(colName);

	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1, std::get<2>(colVal));

	mask[0] = OP{}.template operator() < T, U > (cnst, reinterpret_cast<U*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation filterConstCol: " << colName << ", " << reg << ": " << static_cast<int64_t>(mask[0]) << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterColCol()
{
	auto colNameLeft = arguments.read<std::string>();
	auto colNameRight = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = nullptr;

	if (colNameLeft.front() != '$' &&  colNameRight.front() != '$')
	{
		mask = allocateRegister<int8_t>(reg, 1, true);
		mask[0] = 1;
	}
	else
	{
		loadCol<T>(colNameLeft);
		loadCol<U>(colNameRight);

		mask = allocateRegister<int8_t>(reg, 1, true);
		std::string colPointerNameLeft = getPointerName(colNameLeft);
		std::string colPointerNameRight = getPointerName(colNameRight);

		auto colValLeft = allocatedPointers.at(colPointerNameLeft);
		auto colValRight = allocatedPointers.at(colPointerNameRight);
		mask[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colValLeft))[0], reinterpret_cast<U*>(std::get<0>(colValRight))[0]);
	}

	std::cout << "Where evaluation filterColCol: " << colNameLeft << ", " << colNameRight << ", " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::filterConstConst()
{
	T constLeft = arguments.read<T>();
	U constRight = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	int8_t * mask = allocateRegister<int8_t>(reg, 1, false);
	mask[0] = OP{}.template operator() < T, U > (constLeft, constRight);

	std::cout << "Where evaluation filterConstConst: " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::logicalColConst()
{
	auto colName = arguments.read<std::string>();
	U cnst = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	loadCol<T>(colName);

	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1, std::get<2>(colVal));

	mask[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colVal))[0], cnst);

	std::cout << "Where evaluation logicalConstCol: " << colName << ", " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::logicalConstCol()
{
	T cnst = arguments.read<T>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	loadCol<U>(colName);

	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);

	int8_t * mask = allocateRegister<int8_t>(reg, 1, std::get<2>(colVal));

	mask[0] = OP{}.template operator() < T, U > (cnst, reinterpret_cast<U*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation logicalColConst: " << colName << ", " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::logicalColCol()
{
	auto colNameLeft = arguments.read<std::string>();
	auto colNameRight = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int8_t* mask = nullptr;

	if (colNameLeft.front() != '$' &&  colNameRight.front() != '$')
	{

		mask = allocateRegister<int8_t>(reg, 1, true);
		mask[0] = 1;
	}
	else
	{
		loadCol<T>(colNameLeft);
		loadCol<U>(colNameRight);

		mask = allocateRegister<int8_t>(reg, 1, true);
		std::string colPointerNameLeft = getPointerName(colNameLeft);
		std::string colPointerNameRight = getPointerName(colNameRight);
		auto colValLeft = allocatedPointers.at(colPointerNameLeft);
		auto colValRight = allocatedPointers.at(colPointerNameRight);
		mask[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colValLeft))[0], reinterpret_cast<U*>(std::get<0>(colValRight))[0]);
	}

	std::cout << "Where evaluation logicalColCol: " << colNameLeft << ", " << colNameRight << ", " << reg << ": " << mask[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::logicalConstConst()
{
	T constLeft = arguments.read<T>();
	U constRight = arguments.read<U>();
	auto reg = arguments.read<std::string>();


	int8_t * mask = allocateRegister<int8_t>(reg, 1, false);
	mask[0] = OP{}.template operator() < T, U > (constLeft, constRight);

	std::cout << "Where evaluation logicalConstConst: " << reg << ": " << mask[0] << std::endl;

	return 0;
}
