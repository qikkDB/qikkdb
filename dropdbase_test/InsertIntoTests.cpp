#include "gtest/gtest.h"
#include "../dropdbase/DataType.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"

TEST(InsertIntoTests, InsertIntoCorrect)
{
/*	int blockSize = 1 << 5;

	std::vector<std::string> tableNames = { {"TableA"}};
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, blockSize, true, tableNames, columnTypes);
	Database::AddToInMemoryDatabaseList(database);

	GpuSqlCustomParser parser(database, "INSERT INTO TableA (colInteger1, colLong1, colFloat1, colPolygon1, colPoint1) VALUES (500,20000000, 2.5, POLYGON((20 15, 11 12, 20 15),(21 30, 35 36, 30 20, 21 30),(61 80,90 89,112 110, 61 80)), POINT(2 5));");
	parser.parse();
	auto& table = database->GetTables().at("TableA");
	
	std::vector<int32_t> dataInIntBlock;
	std::vector<int64_t> dataInLongBlock;
	std::vector<float> dataInFloatBlock;
	std::vector<double> dataInDoubleBlock;
	std::vector<ColmnarDB::Types::Point> dataInPointBlock;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataInPolygonBlock;
	std::vector<std::string> dataInStringBlock;

	for (auto &block : dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("colInteger1").get())->GetBlocksList())
	{
		for (auto &entry : block->GetData())
		{
			dataInIntBlock.push_back(entry);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("colLong1").get())->GetBlocksList())
	{
		for (auto &entry : block->GetData())
		{
			dataInLongBlock.push_back(entry);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("colFloat1").get())->GetBlocksList())
	{
		for (auto &entry : block->GetData())
		{
			dataInFloatBlock.push_back(entry);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("colPoint1").get())->GetBlocksList())
	{
		for (auto &entry : block->GetData())
		{
			dataInPointBlock.push_back(entry);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("colPolygon1").get())->GetBlocksList())
	{
		for (auto &entry : block->GetData())
		{
			dataInPolygonBlock.push_back(entry);
		}
	}

	ColmnarDB::Types::Point point = PointFactory::FromWkt("POINT(10.11 11.1)");
	ColmnarDB::Types::ComplexPolygon polygon = ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))");

	for (int i = 0; i < blockSize; i++)
	{
		ASSERT_EQ(1, dataInIntBlock[i]);
		ASSERT_EQ(static_cast<int64_t>(pow(10, 18)), dataInLongBlock[i]);
		ASSERT_FLOAT_EQ((float) 0.1111, dataInFloatBlock[i]);
		ASSERT_EQ(PointFactory::WktFromPoint(point), PointFactory::WktFromPoint(dataInPointBlock[i]));
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(polygon), ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[i]));
	}

	ColmnarDB::Types::Point addedPoint = PointFactory::FromWkt("POINT(2 5)");
	ColmnarDB::Types::ComplexPolygon addedPolygon = ComplexPolygonFactory::FromWkt("POLYGON((20 15, 11 12, 20 15), (21 30, 35 36, 30 20, 21 30), (61 80, 90 89, 112 110, 61 80))");

	ASSERT_EQ(500, dataInIntBlock[blockSize]);
	ASSERT_EQ(20000000, dataInLongBlock[blockSize]);
	ASSERT_FLOAT_EQ((float)2.5, dataInFloatBlock[blockSize]);
	ASSERT_EQ(PointFactory::WktFromPoint(addedPoint), PointFactory::WktFromPoint(dataInPointBlock[blockSize]));
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(addedPolygon), ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[blockSize]));*/
}

TEST(InsertIntoTests, InsertIntoTableNotFound)
{
	/*int blockSize = 1 << 5;

	std::vector<std::string> tableNames = { {"TableA"} };
	std::vector<DataType> columnTypes = { {COLUMN_INT}, {COLUMN_LONG}, {COLUMN_FLOAT}, {COLUMN_POLYGON}, {COLUMN_POINT} };
	std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, blockSize, true, tableNames, columnTypes);
	Database::AddToInMemoryDatabaseList(database);
	
		GpuSqlCustomParser parser(database, "INSERT INTO TableB (colInteger1, colLong1, colFloat1, colPolygon1, colPoint1) VALUES (500,20000000, 2.5, POLYGON((20 15, 11 12, 20 15),(21 30, 35 36, 30 20, 21 30),(61 80,90 89,112 110, 61 80)), POINT(2 5));");
		
	EXPECT_THROW({ try {parser.parse();
	}
	catch (const std::length_error& expected) {
		EXPECT_STREQ("Table was not found in FROM clause.", expected.what());
		throw;
	} }, std::length_error);*/
}