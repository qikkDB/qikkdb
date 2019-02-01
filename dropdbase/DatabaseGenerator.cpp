#include "DatabaseGenerator.h"
#include "Table.h"
#include "ColumnBase.h"
#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"

/// <summary>
/// Generate new database with pseudo random or the same data.
/// Database will have as many tables as is the size of tableNames.
/// In each column there is specific amount of pseudo random generated data.
/// If there are more columns of the same type, each column will have a different data according to formula: k % (1024 * typeColumnCount).
/// </summary>
/// <param name="databaseName">Name of the database. There is no default value, must be specified.</param>
/// <param name="blockCount">Number of blocks of data that will be in each column. Default value is set to 2.</param>
/// <param name="blockSize">Length of each block of data. Default value is set to 2^18.</param>
/// <param name="sameDataInBlocks">If set to true, all rows will have the same data (because all blocks will have
/// the same data), according to column types. If set to false, data in blocks (rows) will be different (per block),
/// but all blocks will have the same data. Default value is set to false.</param>
/// <param name="tablesNames">Names of tables and that implies the number of tables. Defailt value is one table with name 'TableA'.</param>
/// <param name="columnsTypes">Types of columns in table and that implies the number of columns in each table. Default value is one column of type int32_t.</param>
std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockSize, bool sameDataInBlocks)
{
	return DatabaseGenerator::GenerateDatabase(databaseName, blockCount, blockSize, sameDataInBlocks, std::nullopt, std::nullopt);
}

/// <summary>
/// Generate new database with pseudo random or the same data.
/// Database will have as many tables as is the size of tableNames.
/// In each column there is specific amount of pseudo random generated data.
/// If there are more columns of the same type, each column will have a different data according to formula: k % (1024 * typeColumnCount).
/// </summary>
/// <param name="databaseName">Name of the database. There is no default value, must be specified.</param>
/// <param name="blockCount">Number of blocks of data that will be in each column. Default value is set to 2.</param>
/// <param name="blockSize">Length of each block of data. Default value is set to 2^18.</param>
/// <param name="sameDataInBlocks">If set to true, all rows will have the same data (because all blocks will have
/// the same data), according to column types. If set to false, data in blocks (rows) will be different (per block),
/// but all blocks will have the same data. Default value is set to false.</param>
/// <param name="tablesNames">Names of tables and that implies the number of tables. Defailt value is one table with name 'TableA'.</param>
/// <param name="columnsTypes">Types of columns in table and that implies the number of columns in each table. Default value is one column of type int32_t.</param>
std::shared_ptr<Database> DatabaseGenerator::GenerateDatabase(const char * databaseName, int blockCount, int blockSize, bool sameDataInBlocks, std::optional<const std::vector<std::string>> tablesNames, std::optional<const std::vector<DataType>> columnsTypes)
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

	//column counts - used when there are more columns of the same type, then this is used for making data in those columns different
	//NOTE: There has to be block size big enough, so that the change in data is visible, because algorithm is: k % (1024 * intColumnCount)
	int integerColumnCount = 0;
	int longColumnCount = 0;
	int floatColumnCount = 0;
	int doubleColumnCount = 0;
	int pointColumnCount = 0;
	int polygonColumnCount = 0;
	int stringColumnCount = 0;
	
	auto database = std::make_shared<Database>(databaseName, blockSize);

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
				integerColumnCount++;
				table.CreateColumn((std::string("colInteger") + std::to_string(integerColumnCount)).c_str(), COLUMN_INT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<int32_t>&>(*columns.at((std::string("colInteger") + std::to_string(integerColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<int32_t> integerData;

					for (int k = 0; k < blockSize; k++)
					{
						integerData.push_back(sameDataInBlocks ? 1 : k % (1024 * integerColumnCount));
					}
					column.AddBlock(integerData);
				}

				break;
			}
			case COLUMN_LONG:
			{
				longColumnCount++;
				table.CreateColumn((std::string("colLong") + std::to_string(longColumnCount)).c_str(), COLUMN_LONG);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<int64_t>&>(*columns.at((std::string("colLong") + std::to_string(longColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<int64_t> longData;

					for (int k = 0; k < blockSize; k++)
					{
						longData.push_back(sameDataInBlocks ? 10^18 : 2*(10^18) + k % (1024 * longColumnCount));
					}
					column.AddBlock(longData);
				}

				break;
			}

			case COLUMN_FLOAT:
			{
				floatColumnCount++;
				table.CreateColumn((std::string("colFloat") + std::to_string(floatColumnCount)).c_str(), COLUMN_FLOAT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<float>&>(*columns.at((std::string("colFloat") + std::to_string(floatColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<float> floatData;

					for (int k = 0; k < blockSize; k++)
					{
						floatData.push_back(sameDataInBlocks ? (float) 0.1111 : (float)(k % (1024 * floatColumnCount) + 0.1111));
					}
					column.AddBlock(floatData);
				}

				break;
			}

			case COLUMN_DOUBLE:
			{
				doubleColumnCount++;
				table.CreateColumn((std::string("colDouble") + std::to_string(doubleColumnCount)).c_str(), COLUMN_DOUBLE);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<double>&>(*columns.at((std::string("colDouble") + std::to_string(doubleColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<double> doubleData;

					for (int k = 0; k < blockSize; k++)
					{
						doubleData.push_back(sameDataInBlocks ? 0.1111111 : k % (1024 * doubleColumnCount) + 0.1111111);
					}
					column.AddBlock(doubleData);
				}

				break;
			}

			case COLUMN_POINT:
			{
				pointColumnCount++;
				table.CreateColumn((std::string("colPoint") + std::to_string(pointColumnCount)).c_str(), COLUMN_POINT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*columns.at((std::string("colPoint") + std::to_string(pointColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<ColmnarDB::Types::Point> pointData;

					for (int k = 0; k < blockSize; k++)
					{
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(10.11 11.1)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(k % (1024 * pointColumnCount) + 200.2222) +
							" " + std::to_string(k % (1024 * pointColumnCount) + 250) + ")"));
					}
					column.AddBlock(pointData);
				}

				break;
			}

			case COLUMN_POLYGON:
			{
				polygonColumnCount++;
				table.CreateColumn((std::string("colPolygon") + std::to_string(polygonColumnCount)).c_str(), COLUMN_POLYGON);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*columns.at((std::string("colPolygon") + std::to_string(polygonColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<ColmnarDB::Types::ComplexPolygon> polygonData;

					for (int k = 0; k < blockSize; k++)
					{
						polygonData.push_back(sameDataInBlocks ? ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") : 
							ComplexPolygonFactory::FromWkt(std::string("POLYGON((10 11, ") + std::to_string(k % (1024 * polygonColumnCount)) + " " + std::to_string(k % (1024 * polygonColumnCount)) + ", 10 11),(21 30, " + std::to_string(k % (1024 * polygonColumnCount) + 25.1111)
								+ " " + std::to_string(k % (1024 * polygonColumnCount) + 26.1111) + ", " + std::to_string(k % (1024 * polygonColumnCount) + 28) + " " + std::to_string(k % (1024 * polygonColumnCount) + 29) + ", 21 30))"));
					}
					column.AddBlock(polygonData);
				}

				break;
			}

			case COLUMN_STRING:
			{
				stringColumnCount++;
				table.CreateColumn((std::string("colString") + std::to_string(stringColumnCount)).c_str(), COLUMN_STRING);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<std::string>&>(*columns.at((std::string("colString") + std::to_string(stringColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<std::string> stringData;

					for (int k = 0; k < blockSize; k++)
					{
						stringData.push_back(sameDataInBlocks ? "Word1enD-1" : "Word" + std::to_string(k % (1024 * stringColumnCount)));
					}
					column.AddBlock(stringData);
				}

				break;
			}

			default:
			{
				integerColumnCount++;
				table.CreateColumn((std::string("colInteger") + std::to_string(integerColumnCount)).c_str(), COLUMN_INT);
				auto& columns = table.GetColumns();
				auto& column = dynamic_cast<ColumnBase<int32_t>&>(*columns.at((std::string("colInteger") + std::to_string(integerColumnCount)).c_str()));

				for (int i = 0; i < blockCount; i++)
				{
					std::vector<int32_t> integerData;

					for (int k = 0; k < blockSize; k++)
					{
						integerData.push_back(sameDataInBlocks ? 1 : k % (1024 * integerColumnCount));
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
