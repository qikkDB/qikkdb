#include "DatabaseGenerator.h"
#include "Table.h"
#include "ColumnBase.h"
#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"

std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockLenght, bool sameDataInBlocks)
{
	return DatabaseGenerator::GenerateDatabase(databaseName, blockCount, blockLenght, sameDataInBlocks, std::nullopt, std::nullopt);
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
					std::vector<int64_t> longData;

					for (int k = 0; k < blockLenght; k++)
					{
						longData.push_back(sameDataInBlocks ? 1000000000000000000 : 2000000000000000000 + k % 1024);
					}
					column.AddBlock(longData);
				}

				break;
			}

			case COLUMN_FLOAT:
			{
				table.CreateColumn("colFloat", COLUMN_FLOAT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<float>&>(*columns.at("colFloat"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<float> floatData;

					for (int k = 0; k < blockLenght; k++)
					{
						floatData.push_back(sameDataInBlocks ? (float) 0.1111 : (float)(k % 1024 + 0.1111));
					}
					column.AddBlock(floatData);
				}

				break;
			}

			case COLUMN_DOUBLE:
			{
				table.CreateColumn("colDouble", COLUMN_DOUBLE);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<double>&>(*columns.at("colDouble"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<double> doubleData;

					for (int k = 0; k < blockLenght; k++)
					{
						doubleData.push_back(sameDataInBlocks ? 0.1111111 : k % 1024 + 0.1111111);
					}
					column.AddBlock(doubleData);
				}

				break;
			}

			case COLUMN_POINT:
			{
				table.CreateColumn("colPoint", COLUMN_POINT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*columns.at("colPoint"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<ColmnarDB::Types::Point> pointData;

					for (int k = 0; k < blockLenght; k++)
					{
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(10.11 11.1)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(k % 1024 + 200.2222) +
							" " + std::to_string(k % 1024 + 250) + ")"));
					}
					column.AddBlock(pointData);
				}

				break;
			}

			case COLUMN_POLYGON:
			{
				table.CreateColumn("colPolygon", COLUMN_POLYGON);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*columns.at("colPoygon"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<ColmnarDB::Types::ComplexPolygon> polygonData;

					for (int k = 0; k < blockLenght; k++)
					{
						polygonData.push_back(sameDataInBlocks ? ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") : 
							ComplexPolygonFactory::FromWkt(std::string("POLYGON((10 11, ") + std::to_string(k % 1024) + " " + std::to_string(k % 1024) + ", 10 11),(21 30, " + std::to_string(k % 1024 + 25.1111)
								+ " " + std::to_string(k % 1024 + 26.1111) + ", " + std::to_string(k % 1024 + 28) + " " + std::to_string(k % 1024 + 29) + ", 21 30))"));
					}
					column.AddBlock(polygonData);
				}

				break;
			}

			case COLUMN_STRING:
			{
				table.CreateColumn("colString", COLUMN_STRING);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<std::string>&>(*columns.at("colString"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<std::string> stringData;

					for (int k = 0; k < blockLenght; k++)
					{
						stringData.push_back(sameDataInBlocks ? "Word1enD-1" : "Word" + std::to_string(k % 1024));
					}
					column.AddBlock(stringData);
				}

				break;
			}

			case COLUMN_BOOL:
			{
				table.CreateColumn("colBool", COLUMN_BOOL);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<bool>&>(*columns.at("colBool"));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<bool> boolData;

					for (int k = 0; k < blockLenght; k++)
					{
						boolData.push_back(sameDataInBlocks ? 0 : k % 2);
					}
					column.AddBlock(boolData);
				}

				break;
			}

			default:
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
			}
		}

	}

	return database;
}
