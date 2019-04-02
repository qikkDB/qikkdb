#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "GpuSqlDispatcherVMFunctions.h"
#include <tuple>

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}


template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPUFilter::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPULogic::colConst<OP, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPULogic::constCol<OP, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Logical: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);

	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPULogic::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}

	return 0;
}


template<typename T>
int32_t GpuSqlDispatcher::logicalNotCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPULogic::not_col<int8_t, T>(mask, reinterpret_cast<T*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::logicalNotConst()
{
	return 0;
}
