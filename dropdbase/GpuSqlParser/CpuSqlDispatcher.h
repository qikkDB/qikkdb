#pragma once

#include <memory>
#include <array>
#include <tuple>
#include <string>
#include <unordered_map>
#include "MemoryStream.h"
#include "../DataType.h"
#include "../Database.h"
#include "../BlockBase.h"
#include "../ColumnBase.h"
#include "ParserExceptions.h"

class CpuSqlDispatcher
{
private:
	typedef int32_t(CpuSqlDispatcher::*CpuDispatchFunction)();
	std::vector<CpuDispatchFunction> cpuDispatcherFunctions;
	const std::shared_ptr<Database> &database;
	int32_t blockIndex;
	int64_t whereResult;
	MemoryStream arguments;
	int32_t instructionPointer;

	std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t, bool>> allocatedPointers;
	bool isRegisterAllocated(std::string& reg);
	std::pair<std::string, std::string> splitColumnName(const std::string& name);
	std::pair<std::string, std::string> getPointerNames(const std::string & colName);

	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseOrFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseAndFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseXorFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseLeftShiftFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseRightShiftFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logarithmFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> arctangent2Functions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> powerFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rootFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> pointFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> leftFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rightFunctions;

	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> yearFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> monthFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> dayFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> hourFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> minuteFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> secondFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> logicalNotFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> minusFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> absoluteFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> sineFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> cosineFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> tangentFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> cotangentFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> arcsineFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> arccosineFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> arctangentFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> logarithm10Functions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> logarithmNaturalFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> exponentialFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> squareRootFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> squareFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> signFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> roundFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> ceilFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> floorFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> ltrimFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> rtrimFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> lowerFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> upperFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> reverseFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE> lenFunctions;
	static CpuDispatchFunction nullFunction;

	static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> whereResultFunctions;

public:
	CpuSqlDispatcher(const std::shared_ptr<Database> &database);
	void addBinaryOperation(DataType left, DataType right, const std::string& op);
	void addUnaryOperation(DataType type, const std::string & op);
	void addWhereResultFunction(DataType dataType);
	int64_t execute(int32_t index);
	void copyExecutionDataTo(CpuSqlDispatcher& other);

	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size, bool resultColColOperation)
	{
		void* allocatedMemory = operator new(size * sizeof(T));
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(allocatedMemory), size, resultColColOperation) });
		return reinterpret_cast<T*>(allocatedMemory);
	}

	template<typename T>
	T getBlockMin(const std::string& tableName, const std::string& columnName)
	{
		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(tableName).GetColumns().find(columnName)->second.get());
		auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex]);

		return block->GetMin();
	}

	template<typename T>
	T getBlockMax(const std::string& tableName, const std::string& columnName)
	{
		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(tableName).GetColumns().find(columnName)->second.get());
		auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex]);

		return block->GetMax();
	}

	~CpuSqlDispatcher()
	{
		for (auto& pointer : allocatedPointers)
		{
			operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
		}

		allocatedPointers.clear();
	}

	template<typename T>
	int32_t loadCol(std::string& colName)
	{
		if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
		{
			std::string tableName;
			std::string columnName;

			std::tie(tableName, columnName) = splitColumnName(colName);
			if (blockIndex >= database->GetTables().at(tableName).GetColumns().at(columnName).get()->GetBlockCount())
			{
				return 1;
			}

			std::string reg_min = colName + "_min";
			std::string reg_max = colName + "_max";
			T * mask_min = allocateRegister<T>(reg_min, 1, false);
			T * mask_max = allocateRegister<T>(reg_max, 1, false);
			mask_min[0] = getBlockMin<T>(tableName, columnName);
			mask_max[0] = getBlockMax<T>(tableName, columnName);
		}
		return 0;
	}

	template<typename OP, typename T, typename U>
	int32_t filterColConst();

	template<typename OP, typename T, typename U>
	int32_t filterConstCol();

	template<typename OP, typename T, typename U>
	int32_t filterColCol();

	template<typename OP, typename T, typename U>
	int32_t filterConstConst();

	template<typename OP>
	int32_t filterStringColConst();

	template<typename OP>
	int32_t filterStringConstCol();

	template<typename OP>
	int32_t filterStringColCol();

	template<typename OP>
	int32_t filterStringConstConst();

	template<typename OP, typename T, typename U>
	int32_t logicalColConst();

	template<typename OP, typename T, typename U>
	int32_t logicalConstCol();

	template<typename OP, typename T, typename U>
	int32_t logicalColCol();

	template<typename OP, typename T, typename U>
	int32_t logicalConstConst();

	template<typename T>
	int32_t logicalNotCol();

	template<typename T>
	int32_t logicalNotConst();

	int32_t nullCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColConst();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstConst();

	template<typename OP>
	int32_t dateExtractCol();

	template<typename OP>
	int32_t dateExtractConst();

	template<typename OP, typename T>
	int32_t arithmeticUnaryCol();

	template<typename OP, typename T>
	int32_t arithmeticUnaryConst();

	template<typename T, typename U>
	int32_t pointColCol();

	template<typename T, typename U>
	int32_t pointColConst();

	template<typename T, typename U>
	int32_t pointConstCol();

	template<typename T, typename U>
	int32_t containsColConst();

	template<typename T, typename U>
	int32_t containsConstCol();

	template<typename T, typename U>
	int32_t containsColCol();

	template <typename T, typename U>
	int32_t containsConstConst();

	template <typename OP, typename T, typename U>
	int32_t polygonOperationColConst();

	template <typename OP, typename T, typename U>
	int32_t polygonOperationConstCol();

	template <typename OP, typename T, typename U>
	int32_t polygonOperationColCol();

	template <typename OP, typename T, typename U>
	int32_t polygonOperationConstConst();

	template<typename OP>
	int32_t stringUnaryCol();

	template<typename OP>
	int32_t stringUnaryConst();

	template<typename OP>
	int32_t stringUnaryNumericCol();

	template<typename OP>
	int32_t stringUnaryNumericConst();

	template<typename OP, typename T>
	int32_t stringBinaryNumericColCol();

	template<typename OP, typename T>
	int32_t stringBinaryNumericColConst();

	template<typename OP, typename T>
	int32_t stringBinaryNumericConstCol();

	template<typename OP, typename T>
	int32_t stringBinaryNumericConstConst();

	template<typename T>
	int32_t whereResultCol() 
	{
		auto colName = arguments.read<std::string>();
		auto regMin = allocatedPointers.at(colName + "_min");
		auto regMax = allocatedPointers.at(colName + "_max");
		T* resultMin = reinterpret_cast<T*>(std::get<0>(regMin));
		T* resultMax = reinterpret_cast<T*>(std::get<0>(regMax));

		int64_t whereResultMin = std::get<2>(regMin) ? 1 : static_cast<int64_t>(resultMin[0]);
		int64_t whereResultMax = std::get<2>(regMax) ? 1 : static_cast<int64_t>(resultMax[0]);

		whereResult = whereResultMin || whereResultMax;

		std::cout << "Where result: " << colName << ", " << whereResult << std::endl;

		return 1;
	}

	template<typename T>
	int32_t whereResultConst()
	{
		T cnst = arguments.read<T>();
		whereResult = static_cast<int64_t>(cnst);

		std::cout << "Where result const: " << whereResult << std::endl;

		return 1;
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		T cnst = arguments.read<T>();
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColConst()
	{
		auto colName = arguments.read<std::string>();
		U cnst = arguments.read<U>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		T cnstLeft = arguments.read<T>();
		U cnstRight = arguments.read<U>();

		throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColCol()
	{
		auto colNameLeft = arguments.read<std::string>();
		auto colNameRight = arguments.read<std::string>();

		throw InvalidOperandsException(colNameLeft, colNameRight, std::string(typeid(OP).name()));
	}

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string(""), std::string("operation"));
	}

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string("operation"));
	}

	template<typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColConst()
	{
		U cnst = arguments.read<U>();
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
	}

	template<typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		auto colName = arguments.read<std::string>();
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
	}

	template<typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColCol()
	{
		auto colNameRight = arguments.read<std::string>();
		auto colNameLeft = arguments.read<std::string>();

		throw InvalidOperandsException(colNameLeft, colNameRight, std::string("operation"));
	}

	template<typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		U cnstRight = arguments.read<U>();
		T cnstLeft = arguments.read<T>();

		throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string("operation"));
	}

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

	template<typename T>
	void addArgument(T argument)
	{
		arguments.insert<T>(argument);
	}
};

template<>
int32_t CpuSqlDispatcher::loadCol<std::string>(std::string& colName);