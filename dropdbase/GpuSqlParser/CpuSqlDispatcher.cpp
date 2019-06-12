#include "CpuSqlDispatcher.h"

CpuSqlDispatcher::CpuSqlDispatcher(const std::shared_ptr<Database> &database) :
	database(database),
	blockIndex(0)
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
