#include "DatabaseGenerator.h"
#include "Table.h"
#include "ColumnBase.h"
#include "BlockBase.h"

std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockLenght, bool sameDataInBlocks)
{
	return DatabaseGenerator::GenerateDatabase(databaseName,blockCount, blockLenght, sameDataInBlocks, std::nullopt, std::nullopt);
}

std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockLenght, bool sameDataInBlocks, std::optional<const std::vector<std::string>> tablesNames, std::optional<const std::vector<DataType>> columnsTypes)
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
	auto database = std::make_shared<Database>(databaseName, blockLenght);

	for (const auto& tableName : tableNames)
	{
		(*database).tables_.insert({ tableName, Table(database, tableName.c_str()) });
		auto& table = database->tables_.at(tableName);

		for (const auto& columnType : columnTypes)
		{
			switch (columnType)
			{
			case COLUMN_INT:
			{
				table.CreateColumn("colInteger", COLUMN_INT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<int32_t>&>(*columns.at("colInteger"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<int32_t> integerData;

					for (int k = 0; k < blockLenght; k++)
					{
						integerData.push_back(sameDataInBlocks ? 1 : k % 1024);
					}
					column.AddBlock(integerData);
				}

				break;
			}
			case COLUMN_LONG:
			{
				table.CreateColumn("colLong", COLUMN_LONG);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<int64_t>&>(*columns.at("colLong"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<int64_t> integerData;

					for (int k = 0; k < blockLenght; k++)
					{
						integerData.push_back(sameDataInBlocks ? 1000000000000000000 : 2000000000000000000 + k % 1024);
					}
					column.AddBlock(integerData);
				}

				break;
			}

			case COLUMN_FLOAT:
			{

			}

			case COLUMN_DOUBLE:

			case COLUMN_POINT:

			case COLUMN_POLYGON:

			case COLUMN_STRING:

			case COLUMN_BOOL:

			default:
				break;
			}
		}

	}

	return database;
}
