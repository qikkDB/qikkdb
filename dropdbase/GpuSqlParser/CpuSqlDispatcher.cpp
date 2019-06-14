#include "CpuSqlDispatcher.h"

std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::whereResultFunctions = { &CpuSqlDispatcher::whereResultConst<int32_t>, &CpuSqlDispatcher::whereResultConst<int64_t>, &CpuSqlDispatcher::whereResultConst<float>, &CpuSqlDispatcher::whereResultConst<double>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &CpuSqlDispatcher::whereResultConst<int8_t>, &CpuSqlDispatcher::whereResultCol<int32_t>, &CpuSqlDispatcher::whereResultCol<int64_t>, &CpuSqlDispatcher::whereResultCol<float>, &CpuSqlDispatcher::whereResultCol<double>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &CpuSqlDispatcher::whereResultCol<int8_t> };


CpuSqlDispatcher::CpuSqlDispatcher(const std::shared_ptr<Database> &database) :
	database(database),
	blockIndex(0),
	instructionPointer(0),
	evaluateMin(false),
	whereResult(1)
{
}

bool CpuSqlDispatcher::isRegisterAllocated(std::string & reg)
{
	return allocatedPointers.find(reg) != allocatedPointers.end();
}

std::pair<std::string, std::string> CpuSqlDispatcher::splitColumnName(const std::string & name)
{
	const size_t separatorPosition = name.find(".");
	const std::string table = name.substr(0, separatorPosition);
	const std::string column = name.substr(separatorPosition + 1);

	return std::make_pair(table, column);
}

void CpuSqlDispatcher::addBinaryOperation(DataType left, DataType right, const std::string & op)
{
	if (op == ">")
	{
		cpuDispatcherFunctions.push_back(greaterFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "<")
	{
		cpuDispatcherFunctions.push_back(lessFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == ">=")
	{
		cpuDispatcherFunctions.push_back(greaterEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "<=")
	{
		cpuDispatcherFunctions.push_back(lessEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "=")
	{
		cpuDispatcherFunctions.push_back(equalFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "!=" || op == "<>")
	{
		cpuDispatcherFunctions.push_back(notEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "AND")
	{
		cpuDispatcherFunctions.push_back(logicalAndFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "OR")
	{
		cpuDispatcherFunctions.push_back(logicalOrFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "*")
	{
		cpuDispatcherFunctions.push_back(mulFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "/")
	{
		cpuDispatcherFunctions.push_back(divFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "+")
	{
		cpuDispatcherFunctions.push_back(addFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "-")
	{
		cpuDispatcherFunctions.push_back(subFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "%")
	{
		cpuDispatcherFunctions.push_back(modFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
}

void CpuSqlDispatcher::addWhereResultFunction(DataType dataType)
{
	cpuDispatcherFunctions.push_back(whereResultFunctions[dataType]);
}

int64_t CpuSqlDispatcher::execute(int32_t index)
{
	blockIndex = index;

	evaluateMin = true;
	int32_t err = 0;
	while (err == 0)
	{
		err = (this->*cpuDispatcherFunctions[instructionPointer++])();
	}
	int64_t whereResultMin = whereResult;
	instructionPointer = 0;
	arguments.reset();

	for (auto& pointer : allocatedPointers)
	{
		operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
	}
	allocatedPointers.clear();

	evaluateMin = false;
	err = 0;
	while (err == 0)
	{
		err = (this->*cpuDispatcherFunctions[instructionPointer++])();
	}
	int64_t whereResultMax = whereResult;
	instructionPointer = 0;
	arguments.reset();

	for (auto& pointer : allocatedPointers)
	{
		operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
	}
	allocatedPointers.clear();

	return whereResultMin || whereResultMax;
}

void CpuSqlDispatcher::copyExecutionDataTo(CpuSqlDispatcher& other)
{
	other.cpuDispatcherFunctions = cpuDispatcherFunctions;
	other.arguments = arguments;
}

std::string CpuSqlDispatcher::getPointerName(const std::string & colName)
{
	return colName + (evaluateMin ? "_min" : "_max");
}
