#include "CpuSqlDispatcher.h"
bool CpuSqlDispatcher::isRegisterAllocated(std::string & reg)
{
	return allocatedPointers.find(reg) != allocatedPointers.end();
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
