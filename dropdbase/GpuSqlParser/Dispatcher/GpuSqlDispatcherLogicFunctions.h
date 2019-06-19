#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "GpuSqlDispatcherVMFunctions.h"
#include <tuple>

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of genric filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

template<typename OP>
int32_t GpuSqlDispatcher::filterStringColConst()
{
	std::string cnst = arguments.read<std::string>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterString: " << colName << " " << reg << std::endl;

	std::tuple<GPUMemory::GPUString, int32_t> column = findStringColumn(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constString = insertConstStringGpu(cnst);
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP>(mask, std::get<0>(column), constString, retSize);
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::filterStringConstCol()
{
	auto colName = arguments.read<std::string>();
	std::string cnst = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterString: " << colName << " " << reg << std::endl;

	std::tuple<GPUMemory::GPUString, int32_t> column = findStringColumn(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constString = insertConstStringGpu(cnst);
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP>(mask, std::get<0>(column), constString, retSize);
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::filterStringColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<std::string>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterString: " << colName << " " << reg << std::endl;

	std::tuple<GPUMemory::GPUString, int32_t> columnLeft = findStringColumn(colNameLeft);
	std::tuple<GPUMemory::GPUString, int32_t> columnRight = findStringColumn(colNameRight);
	int32_t retSize = std::max(std::get<1>(columnLeft), std::get<1>(colRight));

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP>(mask, std::get<0>(columnLeft), std::get<0>(columnRight), retSize);
	}
	return 0;
}


template<typename OP>
int32_t GpuSqlDispatcher::filterStringConstConst()
{
	std::string constRight = arguments.read<std::string>();
	std::string constLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constStringLeft = insertConstStringGpu(constLeft);
		GPUMemory::GPUString constStringRight = insertConstStringGpu(constRight);

		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPUFilter::constConst<OP>(mask, constStringLeft, constStringRight, database->GetBlockSize());
	}
	return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of NOT operation dispatching
/// Implementation for column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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
