#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include "MemoryStream.h"
#include "../DataType.h"
#include "JoinType.h"
#include "../Database.h"
#include "../Table.h"
#include "../ColumnBase.h"
#include "ParserExceptions.h"


class GpuSqlJoinDispatcher
{
public:
	GpuSqlJoinDispatcher(const std::shared_ptr<Database> &database);

	void addJoinFunction(DataType type, std::string op);

	void addJoinDoneFunction();

	template<typename T>
	void addArgument(T argument)
	{
		arguments.insert<T>(argument);
	}

	std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* getJoinIndices();

	void execute();

private:

	typedef int32_t(GpuSqlJoinDispatcher::*DispatchJoinFunction)();
	std::vector<DispatchJoinFunction> dispatcherFunctions;
	MemoryStream arguments;
	const std::shared_ptr<Database> &database;
	int32_t instructionPointer;
	std::unordered_map<std::string, std::vector<std::vector<int32_t>>> joinIndices;

	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinEqualFunctions;
	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinGreaterFunctions;
	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinLessFunctions;
	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinGreaterEqualFunctions;
	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinLessEqualFunctions;
	static std::array<DispatchJoinFunction,
		DataType::DATA_TYPE_SIZE> joinNotEqualFunctions;
	static DispatchJoinFunction joinDoneFunction;

	template<typename OP, typename T>
	int32_t joinCol();

	template<typename OP, typename T>
	int32_t joinConst();

	int32_t joinDone();

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string(""), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string(typeid(OP).name()));
	}

	std::pair<std::string, std::string> splitColumnName(const std::string& colName);
};