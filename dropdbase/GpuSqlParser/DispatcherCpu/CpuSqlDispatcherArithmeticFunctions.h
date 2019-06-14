#include "../CpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include <tuple>

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::arithmeticColConst()
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

	ResultType * result = allocateRegister<ResultType>(reg, 1);
	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);
	result[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colVal))[0], cnst);
	
	std::cout << "Where evaluation arithmeticColConst: " << colName << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::arithmeticConstCol()
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

	ResultType * result = allocateRegister<ResultType>(reg, 1);
	std::string colPointerName = getPointerName(colName);
	auto colVal = allocatedPointers.at(colPointerName);
	result[0] = OP{}.template operator() < T, U > (cnst, reinterpret_cast<U*>(std::get<0>(colVal))[0]);

	std::cout << "Where evaluation arithmeticConstCol: " << colName << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::arithmeticColCol()
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

	ResultType * result = nullptr;

	if (colNameLeft.front() != '$' &&  colNameRight.front() != '$')
	{
		result = allocateRegister<ResultType>(reg, 1);
		result[0] = 1;
	}
	else
	{
		loadCol<T>(colNameLeft);
		loadCol<U>(colNameRight);

		result = allocateRegister<ResultType>(reg, 1);
		std::string colPointerNameLeft = getPointerName(colNameLeft);
		std::string colPointerNameRight = getPointerName(colNameRight);
		auto colValLeft = allocatedPointers.at(colPointerNameLeft);
		auto colValRight = allocatedPointers.at(colPointerNameRight);
		result[0] = OP{}.template operator() < T, U > (reinterpret_cast<T*>(std::get<0>(colValLeft))[0], reinterpret_cast<U*>(std::get<0>(colValRight))[0]);
	}

	std::cout << "Where evaluation arithmeticColCol: " << colNameLeft << ", " << colNameRight << ", " << reg << ": " << result[0] << std::endl;

	return 0;
}

template<typename OP, typename T, typename U>
inline int32_t CpuSqlDispatcher::arithmeticConstConst()
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
	
	ResultType * result = allocateRegister<ResultType>(reg, 1);
	result[0] = OP{}.template operator() < T, U > (constLeft, constRight);

	std::cout << "Where evaluation arithmeticConstConst: " << reg << ": " << result[0] << std::endl;

	return 0;
}