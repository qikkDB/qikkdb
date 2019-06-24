#include "GpuSqlJoinDispatcher.h"
#include "../QueryEngine/GPUCore/GPUFilterConditions.cuh"

GpuSqlJoinDispatcher::GpuSqlJoinDispatcher(const std::shared_ptr<Database>& database) :
	database(database),
	instructionPointer(0)
{
}

void GpuSqlJoinDispatcher::execute()
{
	int32_t err = 0;

	while (err == 0)
	{
		err = (this->*dispatcherFunctions[instructionPointer++])();

		if (err)
		{
			if (err != 1)
			{
				std::cout << "Error occured while producing join indices." << std::endl;
			}
			break;
		}
	}
}

void GpuSqlJoinDispatcher::addJoinFunction(DataType type, std::string op)
{
	if (op == "=")
	{
		dispatcherFunctions.push_back(joinEqualFunctions[type]);
	}
	else if (op == ">")
	{
		dispatcherFunctions.push_back(joinGreaterFunctions[type]);
	}
	else if (op == "<")
	{
		dispatcherFunctions.push_back(joinLessFunctions[type]);
	}
	else if (op == ">=")
	{
		dispatcherFunctions.push_back(joinGreaterEqualFunctions[type]);
	}
	else if (op == "<=")
	{
		dispatcherFunctions.push_back(joinLessEqualFunctions[type]);
	}
	else if (op == "=" || op == "<>")
	{
		dispatcherFunctions.push_back(joinNotEqualFunctions[type]);
	}
}

void GpuSqlJoinDispatcher::addJoinDoneFunction()
{
	dispatcherFunctions.push_back(joinDoneFunction);
}

std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* GpuSqlJoinDispatcher::getJoinIndices()
{
	return &joinIndices;
}

int32_t GpuSqlJoinDispatcher::joinDone()
{
	std::cout << "Join done." << std::endl;
	return 1;
}

std::pair<std::string, std::string> GpuSqlJoinDispatcher::splitColumnName(const std::string& colName)
{
	const size_t splitIdx = colName.find(".");
	const std::string table = colName.substr(0, splitIdx);
	const std::string column = colName.substr(splitIdx + 1);
	return { table, column };
}

