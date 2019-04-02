#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColConst: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);
		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
		}
	}
	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();


	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticConstCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
		}
	}
	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

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

	std::cout << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (groupByColumns.find(colNameRight) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight + "_keys");
			std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (groupByColumns.find(colNameLeft) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
			std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft + "_keys");
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
		std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
		}
	}
	freeColumnIfRegister<T>(colNameLeft);
	freeColumnIfRegister<U>(colNameRight);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point< U>::value, U, void>::type>::type
	>::type ResultType;
	std::cout << "ArithmeticConstConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		ResultType * result = allocateRegister<ResultType>(reg, retSize);
		GPUArithmetic::constConst<OP, ResultType, T, U>(result, constLeft, constRight, retSize);
	}
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::minusCol()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::minusConst()
{
	return 0;
}