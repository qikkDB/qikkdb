#include "DatabaseGenerator.h"
#include "Table.h"

std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockLenght, bool sameDataInBlocks)
{
	return DatabaseGenerator::GenerateDatabase(databaseName,blockCount, blockLenght, sameDataInBlocks, std::nullopt, std::nullopt);
}

std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockLenght, bool sameDataInBlocks, std::optional<const std::vector<std::string>&> tablesNames, std::optional<const std::vector<DataType>&> columnsTypes)
{
	std::vector<std::string> tableNames = { "TableA" };
	std::vector<DataType> columnTypes = { COLUMN_INT };
	if (tablesNames.has_value())
	{
		tableNames = tablesNames.value();
	}

	if (columnsTypes.has_value())
	{
		columnTypes = columnsTypes.value();
	}
	//TODO dorob args
	auto database = std::make_shared<Database>();

	for (const auto& tableName : tableNames)
	{
		Table table(database, tableName.c_str);

	}

	return database;
}
