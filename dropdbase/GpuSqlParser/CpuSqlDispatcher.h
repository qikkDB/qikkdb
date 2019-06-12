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
	MemoryStream arguments;

	std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t, bool>> allocatedPointers;
	bool isRegisterAllocated(std::string& reg);
	std::pair<std::string, std::string> splitColumnName(const std::string& name);

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

public:
	CpuSqlDispatcher(const std::shared_ptr<Database> &database);
	void addBinaryOperation(DataType left, DataType right, const std::string& op);

	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size)
	{
		void* allocatedMemory = operator new(size * sizeof(T));
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(allocatedMemory), size, true) });
		return reinterpret_cast<T*>(allocatedMemory);
	}

	template<typename T>
	T getBlockMin(const std::string& tableName, const std::string& columnName)
	{
		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(tableName).GetColumns().at(columnName));
		auto block = dynamic_cast<BlockBase<T>*>(col);

		return block->GetMin();
	}

	template<typename T>
	T getBlockMax(const std::string& tableName, const std::string& columnName)
	{
		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(tableName).GetColumns().at(columnName));
		auto block = dynamic_cast<BlockBase<T>*>(col);

		return block->GetMax();
	}

	~CpuSqlDispatcher()
	{
		for (auto& pointer : allocatedPointers)
		{
			operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
		}
	}
	template<typename OP, typename T, typename U>
	int32_t filterColConst();

	template<typename OP, typename T, typename U>
	int32_t filterConstCol();

	template<typename OP, typename T, typename U>
	int32_t filterColCol();

	template<typename OP, typename T, typename U>
	int32_t filterConstConst();

	template<typename OP, typename T, typename U>
	int32_t logicalColConst();

	template<typename OP, typename T, typename U>
	int32_t logicalConstCol();

	template<typename OP, typename T, typename U>
	int32_t logicalColCol();

	template<typename OP, typename T, typename U>
	int32_t logicalConstConst();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColConst();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstConst();


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		auto colName = arguments.read<std::string>();
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
		return 0;
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColConst()
	{
		U cnst = arguments.read<U>();
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
		return 0;
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		U cnstRight = arguments.read<U>();
		T cnstLeft = arguments.read<T>();

		throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string(typeid(OP).name()));
		return 0;
	}

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColCol()
	{
		auto colNameRight = arguments.read<std::string>();
		auto colNameLeft = arguments.read<std::string>();

		throw InvalidOperandsException(colNameLeft, colNameRight, std::string(typeid(OP).name()));
		return 0;
	}
};