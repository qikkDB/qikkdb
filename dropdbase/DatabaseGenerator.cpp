#include <cmath>

#include "DatabaseGenerator.h"
#include "Table.h"
#include "ColumnBase.h"
#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include <ctime>

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
		(*database).tables_.emplace(std::make_pair(tableName, Table(database, tableName.c_str())));
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
						if (k % 2)
						{
							integerData.push_back(sameDataInBlocks ? 1 : k % (1024 * integerColumnCount));
						}
						else
						{
							integerData.push_back(sameDataInBlocks ? -1 : (k % (1024 * integerColumnCount)) * -1);
						}
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

						if (longColumnCount % 3 != 0)
						{
							int64_t result = (static_cast<int64_t>(2 * pow(10, k % 19))) + k % (1024 * longColumnCount);

							if (k % 2)
							{
								longData.push_back(sameDataInBlocks ? static_cast<int64_t>(pow(10, k % 19)) : result);
							}
							else
							{
								longData.push_back(sameDataInBlocks ? static_cast<int64_t>(pow(10, k % 19)) * -1 : result * (-1));
							}
						}

						else
						{
							struct tm date;
							date.tm_hour = sameDataInBlocks ? 10 : (k % 24);
							date.tm_min = sameDataInBlocks ? 10 : ((k + 1) % 60);
							date.tm_sec = sameDataInBlocks ? 10 : ((k + 2) % 60);

							date.tm_mday = sameDataInBlocks ? 10 : ((k % 28) + 1);
							date.tm_mon = sameDataInBlocks ? 10 : (k % 12);
							date.tm_year = sameDataInBlocks ? 110 : ((k % 1000) + 100);
#ifdef WIN32 
							const time_t utcTimestamp = _mkgmtime64(&date);
#else
							const time_t utcTimestamp = timegm(&date);
#endif

							longData.push_back(static_cast<int64_t>(utcTimestamp));
						}
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
						if (k % 2)
						{
							floatData.push_back(sameDataInBlocks ? (float) 0.1111 : (float)(k % (1024 * floatColumnCount) + 0.1111));
						}
						else
						{
							floatData.push_back(sameDataInBlocks ? (float)-0.1111 : (float)((k % (1024 * floatColumnCount) + 0.1111) * -1));
						}
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
						if (k % 2)
						{
							doubleData.push_back(sameDataInBlocks ? 0.1111111 : k % (1024 * doubleColumnCount) + 0.1111111);
						}
						else
						{
							doubleData.push_back(sameDataInBlocks ? -0.1111111 : (k % (1024 * doubleColumnCount) + 0.1111111) * -1);
						}
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

					int k = 0;
					while (k + 4 < blockSize)
					{
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(15 15.4)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(15 + 5 * (pointColumnCount - 1) + k * 0.001) +
							" " + std::to_string(15.4 + 5 * (pointColumnCount - 1) + k * 0.001) + ")")); //outside
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(15.5 10.5)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(15.5 + 5 * (pointColumnCount - 1) + k * 0.001) +
							" " + std::to_string(10.5 + 5 * (pointColumnCount - 1) + k * 0.001) + ")")); //on line, so it is inside
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(10.5 10.5)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(10.5 + 5 * (pointColumnCount - 1) + k * 0.001) +
							" " + std::to_string(10.5 + 5 * (pointColumnCount - 1) + k * 0.001) + ")")); //same vertex, so it is inside
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(-30 -30.6)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string((30 + 5 * (pointColumnCount - 1) + k * 0.001) * -1) +
							" " + std::to_string((30.6 + 5 * (pointColumnCount - 1) + k * 0.001) * -1) + ")")); //outside
						k += 4;
					}
					while (k < blockSize)
					{
						pointData.push_back(sameDataInBlocks ? PointFactory::FromWkt("POINT(15 15.4)") : PointFactory::FromWkt(std::string("POINT(") + std::to_string(15 + 5 * pointColumnCount + k * 0.001) +
							" " + std::to_string(15.4 + 5 * (pointColumnCount - 1) + k * 0.001) + ")")); //inside
						k++;
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
						polygonData.push_back(sameDataInBlocks ? ComplexPolygonFactory::FromWkt("POLYGON((-1 1, 2.5 2.5, -1 1),(5 15.5, 10.5 10.5, 20.5 10.5, 25 15.5, 20.5 20, 10.5 20, 5 15.5))") :
							ComplexPolygonFactory::FromWkt(std::string("POLYGON((1 1, 2.5 2.5, 1 1),(") + std::to_string(5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " + std::to_string(15.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(10.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " +
								std::to_string(10.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(20.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " + std::to_string(10.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(25 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " +
								std::to_string(15.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(20.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " + std::to_string(20 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(10.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " +
								std::to_string(20 + 5 * (polygonColumnCount - 1) + k * 0.001) + ", " + std::to_string(5 + 5 * (polygonColumnCount - 1) + k * 0.001) + " " + std::to_string(15.5 + 5 * (polygonColumnCount - 1) + k * 0.001) + "))"));
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
						stringData.push_back(sameDataInBlocks ? "Word1enD1" : "Word" + std::to_string(k % (1024 * stringColumnCount)));
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
						if (k % 2)
						{
							integerData.push_back(sameDataInBlocks ? 1 : k % (1024 * integerColumnCount));
						}
						else
						{
							integerData.push_back(sameDataInBlocks ? -1 : (k % (1024 * integerColumnCount)) * -1);
						}
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