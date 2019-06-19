#include "CpuSqlDispatcher.h"

std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::whereResultFunctions = { &CpuSqlDispatcher::whereResultConst<int32_t>, &CpuSqlDispatcher::whereResultConst<int64_t>, &CpuSqlDispatcher::whereResultConst<float>, &CpuSqlDispatcher::whereResultConst<double>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &CpuSqlDispatcher::whereResultConst<int8_t>, &CpuSqlDispatcher::whereResultCol<int32_t>, &CpuSqlDispatcher::whereResultCol<int64_t>, &CpuSqlDispatcher::whereResultCol<float>, &CpuSqlDispatcher::whereResultCol<double>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &CpuSqlDispatcher::whereResultCol<int8_t> };


CpuSqlDispatcher::CpuSqlDispatcher(const std::shared_ptr<Database> &database) :
	database(database),
	blockIndex(0),
	instructionPointer(0),
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
	else if (op == "|")
	{
		cpuDispatcherFunctions.push_back(bitwiseOrFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "&")
	{
		cpuDispatcherFunctions.push_back(bitwiseAndFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "^")
	{
		cpuDispatcherFunctions.push_back(bitwiseXorFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "<<")
	{
		cpuDispatcherFunctions.push_back(bitwiseLeftShiftFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == ">>")
	{
		cpuDispatcherFunctions.push_back(bitwiseRightShiftFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "POINT")
	{
		cpuDispatcherFunctions.push_back(pointFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "GEO_CONTAINS")
	{
		cpuDispatcherFunctions.push_back(containsFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "GEO_INTERSECT")
	{
		cpuDispatcherFunctions.push_back(intersectFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "GEO_UNION")
	{
		cpuDispatcherFunctions.push_back(unionFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "LOG")
	{
		cpuDispatcherFunctions.push_back(logarithmFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "POW")
	{
		cpuDispatcherFunctions.push_back(powerFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "ROOT")
	{
		cpuDispatcherFunctions.push_back(rootFunctions[left * DataType::DATA_TYPE_SIZE + right]);
	}
	else if (op == "ATAN2")
	{
		cpuDispatcherFunctions.push_back(arctangent2Functions[left * DataType::DATA_TYPE_SIZE + right]);
	}
}

void CpuSqlDispatcher::addUnaryOperation(DataType type, const std::string & op)
{
	if (op == "!")
	{
		cpuDispatcherFunctions.push_back(logicalNotFunctions[type]);
	}
	else if (op == "-")
	{
		cpuDispatcherFunctions.push_back(minusFunctions[type]);
	}
	else if (op == "YEAR")
	{
		cpuDispatcherFunctions.push_back(yearFunctions[type]);
	}
	else if (op == "MONTH")
	{
		cpuDispatcherFunctions.push_back(monthFunctions[type]);
	}
	else if (op == "DAY")
	{
		cpuDispatcherFunctions.push_back(dayFunctions[type]);
	}
	else if (op == "HOUR")
	{
		cpuDispatcherFunctions.push_back(hourFunctions[type]);
	}
	else if (op == "MINUTE")
	{
		cpuDispatcherFunctions.push_back(minuteFunctions[type]);
	}
	else if (op == "SECOND")
	{
		cpuDispatcherFunctions.push_back(secondFunctions[type]);
	}
	else if (op == "ABS")
	{
		cpuDispatcherFunctions.push_back(absoluteFunctions[type]);
	}
	else if (op == "SIN")
	{
		cpuDispatcherFunctions.push_back(sineFunctions[type]);
	}
	else if (op == "COS")
	{
		cpuDispatcherFunctions.push_back(cosineFunctions[type]);
	}
	else if (op == "TAN")
	{
		cpuDispatcherFunctions.push_back(tangentFunctions[type]);
	}
	else if (op == "COT")
	{
		cpuDispatcherFunctions.push_back(cotangentFunctions[type]);
	}
	else if (op == "ASIN")
	{
		cpuDispatcherFunctions.push_back(arcsineFunctions[type]);
	}
	else if (op == "ACOS")
	{
		cpuDispatcherFunctions.push_back(arccosineFunctions[type]);
	}
	else if (op == "ATAN")
	{
		cpuDispatcherFunctions.push_back(arctangentFunctions[type]);
	}
	else if (op == "LOG10")
	{
		cpuDispatcherFunctions.push_back(logarithm10Functions[type]);
	}
	else if (op == "LOG")
	{
		cpuDispatcherFunctions.push_back(logarithmNaturalFunctions[type]);
	}
	else if (op == "EXP")
	{
		cpuDispatcherFunctions.push_back(exponentialFunctions[type]);
	}
	else if (op == "SQRT")
	{
		cpuDispatcherFunctions.push_back(squareRootFunctions[type]);
	}
	else if (op == "SQUARE")
	{
		cpuDispatcherFunctions.push_back(squareFunctions[type]);
	}
	else if (op == "SIGN")
	{
		cpuDispatcherFunctions.push_back(signFunctions[type]);
	}
	else if (op == "ROUND")
	{
		cpuDispatcherFunctions.push_back(roundFunctions[type]);
	}
	else if (op == "FLOOR")
	{
		cpuDispatcherFunctions.push_back(floorFunctions[type]);
	}
	else if (op == "CEIL")
	{
		cpuDispatcherFunctions.push_back(ceilFunctions[type]);
	}
}

void CpuSqlDispatcher::addWhereResultFunction(DataType dataType)
{
	cpuDispatcherFunctions.push_back(whereResultFunctions[dataType]);
}

int64_t CpuSqlDispatcher::execute(int32_t index)
{
	blockIndex = index;

	int32_t err = 0;
	while (err == 0)
	{
		err = (this->*cpuDispatcherFunctions[instructionPointer++])();
	}
	instructionPointer = 0;
	arguments.reset();

	for (auto& pointer : allocatedPointers)
	{
		operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
	}
	allocatedPointers.clear();

	return whereResult;
}

void CpuSqlDispatcher::copyExecutionDataTo(CpuSqlDispatcher& other)
{
	other.cpuDispatcherFunctions = cpuDispatcherFunctions;
	other.arguments = arguments;
}

std::pair<std::string, std::string> CpuSqlDispatcher::getPointerNames(const std::string & colName)
{
	return { colName + "_min", colName + "_max" };
}

