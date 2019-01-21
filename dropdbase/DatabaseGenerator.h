#pragma once
#include "Database.h"
#include "DataType.h"
#include <optional>
#include <vector>
#include <string>

class DatabaseGenerator
{
private:
	DatabaseGenerator() {};
public:
	static std::shared_ptr<Database> GenerateDatabase(const char* databaseName, int blockCount = 2, int blockLenght = 1 << 18, bool sameDataInBlocks = false);
	static std::shared_ptr<Database> GenerateDatabase(const char* databaseName, int blockCount, int blockLenght, bool sameDataInBlocks, std::optional<const std::vector<std::string>&> tablesNames, std::optional<const std::vector<DataType>&> columnsTypes);
};

