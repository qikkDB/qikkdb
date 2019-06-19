#pragma once

#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include <tuple>

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticColConst()
{
	auto colName = arguments.read<std::string>();
	U cnst = arguments.read<U>();
	auto reg = arguments.read<std::string>();

	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

	loadCol<T>(colName);

	std::string colPointerNameMin;
	std::string colPointerNameMax;
	std::tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

	auto colValMin = allocatedPointers.at(colPointerNameMin);
	auto colValMax = allocatedPointers.at(colPointerNameMax);

	ResultType * resultMin = allocateRegister<ResultType>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
	ResultType * resultMax = allocateRegister<ResultType>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);

	resultMin[0] = OP{}.template operator() < ResultType, T, U > (reinterpret_cast<T*>(std::get<0>(colValMin))[0], cnst);
	resultMax[0] = OP{}.template operator() < ResultType, T, U > (reinterpret_cast<T*>(std::get<0>(colValMax))[0], cnst);
	
	std::cout << std::string(typeid(ResultType).name()) << std::endl;
	std::cout << "Where evaluation arithmeticColConstMin: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0] << ", " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticColConstMax: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0] << ", " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticConstCol()
{
	T cnst = arguments.read<T>();
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

	loadCol<U>(colName);

	std::string colPointerNameMin;
	std::string colPointerNameMax;
	std:tie(colPointerNameMin, colPointerNameMax) = getPointerNames(colName);

	auto colValMin = allocatedPointers.at(colPointerNameMin);
	auto colValMax = allocatedPointers.at(colPointerNameMax);

	ResultType * resultMin = allocateRegister<ResultType>(reg + "_min", 1, std::get<2>(colValMin) || !OP::isMonotonous);
	ResultType * resultMax = allocateRegister<ResultType>(reg + "_max", 1, std::get<2>(colValMax) || !OP::isMonotonous);

	resultMin[0] = OP{}.template operator() < ResultType, T, U > (cnst, reinterpret_cast<U*>(std::get<0>(colValMin))[0]);
	resultMax[0] = OP{}.template operator() < ResultType, T, U > (cnst, reinterpret_cast<U*>(std::get<0>(colValMax))[0]);

	std::cout << "Where evaluation arithmeticConstColMin: " << reinterpret_cast<T*>(std::get<0>(colValMin))[0] << ", " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticConstColMax: " << reinterpret_cast<T*>(std::get<0>(colValMax))[0] << ", " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticColCol()
{
	auto colNameLeft = arguments.read<std::string>();
	auto colNameRight = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

	ResultType * resultMin = nullptr;
	ResultType * resultMax = nullptr;

	if (colNameLeft.front() != '$' &&  colNameRight.front() != '$')
	{
		resultMin = allocateRegister<ResultType>(reg + "_min", 1, true);
		resultMax = allocateRegister<ResultType>(reg + "_max", 1, true);
		resultMin[0] = 1;
		resultMax[0] = 1;
	}
	else
	{
		loadCol<T>(colNameLeft);
		loadCol<U>(colNameRight);

		std::string colPointerNameLeftMin;
		std::string colPointerNameLeftMax;
		std::tie(colPointerNameLeftMin, colPointerNameLeftMax) = getPointerNames(colNameLeft);

		std::string colPointerNameRightMin;
		std::string colPointerNameRightMax;
		std::tie(colPointerNameRightMin, colPointerNameRightMax) = getPointerNames(colNameRight);

		auto colValLeftMin = allocatedPointers.at(colPointerNameLeftMin);
		auto colValLeftMax = allocatedPointers.at(colPointerNameLeftMax);

		auto colValRightMin = allocatedPointers.at(colPointerNameRightMin);
		auto colValRightMax = allocatedPointers.at(colPointerNameRightMax);

		resultMin = allocateRegister<ResultType>(reg + "_min", 1, std::get<2>(colValLeftMin) || std::get<2>(colValRightMin));
		resultMax = allocateRegister<ResultType>(reg + "_max", 1, std::get<2>(colValLeftMax) || std::get<2>(colValRightMax));
		
		resultMin[0] = OP{}.template operator() < ResultType, T, U > (reinterpret_cast<T*>(std::get<0>(colValLeftMin))[0], reinterpret_cast<U*>(std::get<0>(colValRightMin))[0]);
		resultMax[0] = OP{}.template operator() < ResultType, T, U > (reinterpret_cast<T*>(std::get<0>(colValLeftMax))[0], reinterpret_cast<U*>(std::get<0>(colValRightMax))[0]);
	}
	std::cout << "Where evaluation arithmeticColCol_min: " << colNameLeft << ", " << colNameRight << ", " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticColCol_max: " << colNameLeft << ", " << colNameRight << ", " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::arithmeticConstConst()
{
	T constLeft = arguments.read<T>();
	U constRight = arguments.read<U>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point< U>::value, U, void>::type>::type
	>::type ResultType;
	
	ResultType * resultMin = allocateRegister<ResultType>(reg + "_min", 1, false);
	ResultType * resultMax = allocateRegister<ResultType>(reg + "_max", 1, false);

	resultMin[0] = OP{}.template operator() < ResultType, T, U > (constLeft, constRight);
	resultMax[0] = OP{}.template operator() < ResultType, T, U > (constLeft, constRight);

	std::cout << "Where evaluation arithmeticConstConst_min: " << reg + "_min" << ": " << resultMin[0] << std::endl;
	std::cout << "Where evaluation arithmeticConstConst_max: " << reg + "_max" << ": " << resultMax[0] << std::endl;

	return 0;
}