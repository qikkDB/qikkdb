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

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	T min = getBlockMin<T>(tableName, columnName);
	T max = getBlockMax<T>(tableName, columnName);

	//TODO there is only implementation with min, but also should work with max
	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);
	ResultType * result = allocateRegister<ResultType>(reg, retSize);
	*result = OP{}.template operator() < T, U > (min, cnst);
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

	std::string tableName;
	std::string columnName;

	std::tie(tableName, columnName) = splitColumnName(colName);

	U min = getBlockMin<U>(tableName, columnName);
	U max = getBlockMax<U>(tableName, columnName);

	//TODO there is only implementation with min, but also should work with max
	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);
	ResultType * result = allocateRegister<ResultType>(reg, retSize);
	//
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

	ResultType * result = allocateRegister<ResultType>(reg, 1);
	*result = 1;

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
	//
	return 0;
}