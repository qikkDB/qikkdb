#include "../dropdbase/BlockBase.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/Table.h"
#include "gtest/gtest.h"

TEST(BlockTests, InsertDataVector)
{
    auto database = std::make_shared<Database>("testDatabase", 1024);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    table.CreateColumn("ColumnLong", COLUMN_LONG);
    table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
    table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
    table.CreateColumn("ColumnPoint", COLUMN_POINT);
    table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
    table.CreateColumn("ColumnString", COLUMN_STRING);

    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& columnLong = table.GetColumns().at("ColumnLong");
    auto& columnFloat = table.GetColumns().at("ColumnFloat");
    auto& columnDouble = table.GetColumns().at("ColumnDouble");
    auto& columnPoint = table.GetColumns().at("ColumnPoint");
    auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
    auto& columnString = table.GetColumns().at("ColumnString");


    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();
    auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock();
    auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock();
    auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock();
    auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock();
    auto& blockPolygon =
        dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock();
    auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock();


    std::vector<int32_t> dataInt;
    std::vector<int64_t> dataLong;
    std::vector<float> dataFloat;
    std::vector<double> dataDouble;
    std::vector<ColmnarDB::Types::Point> dataPoint;
    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
    std::vector<std::string> dataString;

    for (int i = 0; i < 1024; i++)
    {
        dataInt.push_back(i);
        dataLong.push_back(i * 1000000000);
        dataFloat.push_back((float)0.1111 * i);
        dataDouble.push_back((double)i);
        dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
        dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 "
            "80.11,90 89.15,112.12 110, 61 80.11))"));
        dataString.push_back("abc");
    }

    blockInt.InsertData(dataInt);
    blockLong.InsertData(dataLong);
    blockFloat.InsertData(dataFloat);
    blockDouble.InsertData(dataDouble);
    blockPoint.InsertData(dataPoint);
    blockPolygon.InsertData(dataPolygon);
    blockString.InsertData(dataString);

    // Data fit into block
    for (int i = 0; i < 1024; i++)
    {
        ASSERT_EQ(blockInt.GetData()[i], dataInt[i]);
        ASSERT_EQ(blockLong.GetData()[i], dataLong[i]);
        ASSERT_EQ(blockFloat.GetData()[i], dataFloat[i]);
        ASSERT_EQ(blockDouble.GetData()[i], dataDouble[i]);
        ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.GetData()[i]),
                  PointFactory::WktFromPoint(dataPoint[i]));
        ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.GetData()[i]),
                  ComplexPolygonFactory::WktFromPolygon(dataPolygon[i]));
        ASSERT_EQ(blockString.GetData()[i], dataString[i]);
    }

    // Data do not fit into block
    EXPECT_THROW(
        {
            try
            {
                blockInt.InsertData(dataInt);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockLong.InsertData(dataLong);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockFloat.InsertData(dataFloat);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockDouble.InsertData(dataDouble);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockPoint.InsertData(dataPoint);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockPolygon.InsertData(dataPolygon);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);

    EXPECT_THROW(
        {
            try
            {
                blockString.InsertData(dataString);
            }
            catch (const std::length_error& expected)
            {
                EXPECT_STREQ("Attempted to insert data larger than remaining block size", expected.what());
                throw;
            }
        },
        std::length_error);
}

TEST(BlockTests, FindIndexAndRange)
{
    auto database = std::make_shared<Database>("testDatabase", 8);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();

    blockInt.InsertDataOnSpecificPosition(0, 1);
    blockInt.InsertDataOnSpecificPosition(1, 1);
    blockInt.InsertDataOnSpecificPosition(2, 1);
    blockInt.InsertDataOnSpecificPosition(3, 1);
    blockInt.InsertDataOnSpecificPosition(4, 2);
    blockInt.InsertDataOnSpecificPosition(5, 2);
    blockInt.InsertDataOnSpecificPosition(6, 2);
    blockInt.InsertDataOnSpecificPosition(7, 2);

    int index;
    int range;
    bool reachEnd;

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 2, 0);
    ASSERT_EQ(index, 0);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 7, 0);
    ASSERT_EQ(index, 0);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 7, 1);
    ASSERT_EQ(index, 0);
    ASSERT_EQ(range, 4);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(6, 1, 1);
    ASSERT_EQ(index, 6);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 2, 2);
    ASSERT_EQ(index, 2);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 7, 2);
    ASSERT_EQ(index, 4);
    ASSERT_EQ(range, 3);
    ASSERT_EQ(reachEnd, false);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 8, 2);
    ASSERT_EQ(index, 4);
    ASSERT_EQ(range, 4);
    ASSERT_EQ(reachEnd, true);

    std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 9, 2);
    ASSERT_EQ(index, 4);
    ASSERT_EQ(range, 4);
    ASSERT_EQ(reachEnd, true);
}

TEST(BlockTests, FindIndexAndRange_valuesInsertedProgressively)
{
    auto database = std::make_shared<Database>("testDatabase", 4);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();

    blockInt.InsertDataOnSpecificPosition(0, 2);

	int index;
    int range;
    bool reachEnd;

	std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 10, 1);
    ASSERT_EQ(index, 0);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, false);

	blockInt.InsertDataOnSpecificPosition(0, 1);
	
	std::tie(index, range, reachEnd) = blockInt.FindIndexAndRange(0, 10, 5);
    ASSERT_EQ(index, 2);
    ASSERT_EQ(range, 0);
    ASSERT_EQ(reachEnd, true);
}

TEST(BlockTests, InserDatatOnSpecificPosition)
{
    auto database = std::make_shared<Database>("testDatabase", 20);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();

    blockInt.InsertDataOnSpecificPosition(0, 2);
    ASSERT_EQ(blockInt.GetData()[0], 2);

    blockInt.InsertDataOnSpecificPosition(1, 30);
    ASSERT_EQ(blockInt.GetData()[0], 2);
    ASSERT_EQ(blockInt.GetData()[1], 30);

    blockInt.InsertDataOnSpecificPosition(1, 5);
    ASSERT_EQ(blockInt.GetData()[0], 2);
    ASSERT_EQ(blockInt.GetData()[1], 5);
    ASSERT_EQ(blockInt.GetData()[2], 30);

    blockInt.InsertDataOnSpecificPosition(0, 1);
    ASSERT_EQ(blockInt.GetData()[0], 1);
    ASSERT_EQ(blockInt.GetData()[1], 2);
    ASSERT_EQ(blockInt.GetData()[2], 5);
    ASSERT_EQ(blockInt.GetData()[3], 30);

    blockInt.InsertDataOnSpecificPosition(1, 50);
    ASSERT_EQ(blockInt.GetData()[0], 1);
    ASSERT_EQ(blockInt.GetData()[1], 50);
    ASSERT_EQ(blockInt.GetData()[2], 2);
    ASSERT_EQ(blockInt.GetData()[3], 5);
    ASSERT_EQ(blockInt.GetData()[4], 30);

    blockInt.InsertDataOnSpecificPosition(7, 50);
    ASSERT_EQ(blockInt.GetData()[0], 1);
    ASSERT_EQ(blockInt.GetData()[1], 50);
    ASSERT_EQ(blockInt.GetData()[2], 2);
    ASSERT_EQ(blockInt.GetData()[3], 5);
    ASSERT_EQ(blockInt.GetData()[4], 30);
    ASSERT_EQ(blockInt.GetData()[7], 50);
}

TEST(BlockTests, IsFull)
{
    auto database = std::make_shared<Database>("testDatabase", 1024);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    table.CreateColumn("ColumnLong", COLUMN_LONG);
    table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
    table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
    table.CreateColumn("ColumnPoint", COLUMN_POINT);
    table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
    table.CreateColumn("ColumnString", COLUMN_STRING);

    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& columnLong = table.GetColumns().at("ColumnLong");
    auto& columnFloat = table.GetColumns().at("ColumnFloat");
    auto& columnDouble = table.GetColumns().at("ColumnDouble");
    auto& columnPoint = table.GetColumns().at("ColumnPoint");
    auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
    auto& columnString = table.GetColumns().at("ColumnString");

    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();
    auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock();
    auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock();
    auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock();
    auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock();
    auto& blockPolygon =
        dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock();
    auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock();

    ASSERT_FALSE(blockInt.IsFull());
    ASSERT_FALSE(blockLong.IsFull());
    ASSERT_FALSE(blockFloat.IsFull());
    ASSERT_FALSE(blockDouble.IsFull());
    ASSERT_FALSE(blockPoint.IsFull());
    ASSERT_FALSE(blockPolygon.IsFull());
    ASSERT_FALSE(blockString.IsFull());

    std::vector<int32_t> dataInt;
    std::vector<int64_t> dataLong;
    std::vector<float> dataFloat;
    std::vector<double> dataDouble;
    std::vector<ColmnarDB::Types::Point> dataPoint;
    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
    std::vector<std::string> dataString;

    for (int i = 0; i < database->GetBlockSize() / 2; i++)
    {
        dataInt.push_back(i);
        dataLong.push_back(i * 1000000000);
        dataFloat.push_back((float)0.1111 * i);
        dataDouble.push_back((double)i);
        dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
        dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 "
            "80.11,90 89.15,112.12 110, 61 80.11))"));
        dataString.push_back("abc");
    }

    blockInt.InsertData(dataInt);
    blockLong.InsertData(dataLong);
    blockFloat.InsertData(dataFloat);
    blockDouble.InsertData(dataDouble);
    blockPoint.InsertData(dataPoint);
    blockPolygon.InsertData(dataPolygon);
    blockString.InsertData(dataString);

    ASSERT_FALSE(blockInt.IsFull());
    ASSERT_FALSE(blockLong.IsFull());
    ASSERT_FALSE(blockFloat.IsFull());
    ASSERT_FALSE(blockDouble.IsFull());
    ASSERT_FALSE(blockPoint.IsFull());
    ASSERT_FALSE(blockPolygon.IsFull());
    ASSERT_FALSE(blockString.IsFull());

    blockInt.InsertData(dataInt);
    blockLong.InsertData(dataLong);
    blockFloat.InsertData(dataFloat);
    blockDouble.InsertData(dataDouble);
    blockPoint.InsertData(dataPoint);
    blockPolygon.InsertData(dataPolygon);
    blockString.InsertData(dataString);

    ASSERT_TRUE(blockInt.IsFull());
    ASSERT_TRUE(blockLong.IsFull());
    ASSERT_TRUE(blockFloat.IsFull());
    ASSERT_TRUE(blockDouble.IsFull());
    ASSERT_TRUE(blockPoint.IsFull());
    ASSERT_TRUE(blockPolygon.IsFull());
    ASSERT_TRUE(blockString.IsFull());
}

TEST(BlockTests, EmptyBlockSpace)
{
    auto database = std::make_shared<Database>("testDatabase", 1024);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    table.CreateColumn("ColumnLong", COLUMN_LONG);
    table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
    table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
    table.CreateColumn("ColumnPoint", COLUMN_POINT);
    table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
    table.CreateColumn("ColumnString", COLUMN_STRING);

    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& columnLong = table.GetColumns().at("ColumnLong");
    auto& columnFloat = table.GetColumns().at("ColumnFloat");
    auto& columnDouble = table.GetColumns().at("ColumnDouble");
    auto& columnPoint = table.GetColumns().at("ColumnPoint");
    auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
    auto& columnString = table.GetColumns().at("ColumnString");

    auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();
    auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock();
    auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock();
    auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock();
    auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock();
    auto& blockPolygon =
        dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock();
    auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock();

    ASSERT_EQ(blockInt.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockLong.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockFloat.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockDouble.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockPoint.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockPolygon.EmptyBlockSpace(), database->GetBlockSize());
    ASSERT_EQ(blockString.EmptyBlockSpace(), database->GetBlockSize());

    std::vector<int32_t> dataInt;
    std::vector<int64_t> dataLong;
    std::vector<float> dataFloat;
    std::vector<double> dataDouble;
    std::vector<ColmnarDB::Types::Point> dataPoint;
    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
    std::vector<std::string> dataString;

    for (int i = 0; i < database->GetBlockSize() / 2; i++)
    {
        dataInt.push_back(i);
        dataLong.push_back(i * 1000000000);
        dataFloat.push_back((float)0.1111 * i);
        dataDouble.push_back((double)i);
        dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
        dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 "
            "80.11,90 89.15,112.12 110, 61 80.11))"));
        dataString.push_back("abc");
    }

    blockInt.InsertData(dataInt);
    blockLong.InsertData(dataLong);
    blockFloat.InsertData(dataFloat);
    blockDouble.InsertData(dataDouble);
    blockPoint.InsertData(dataPoint);
    blockPolygon.InsertData(dataPolygon);
    blockString.InsertData(dataString);

    ASSERT_EQ(blockInt.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockLong.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockFloat.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockDouble.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockPoint.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockPolygon.EmptyBlockSpace(), database->GetBlockSize() / 2);
    ASSERT_EQ(blockString.EmptyBlockSpace(), database->GetBlockSize() / 2);

    blockInt.InsertData(dataInt);
    blockLong.InsertData(dataLong);
    blockFloat.InsertData(dataFloat);
    blockDouble.InsertData(dataDouble);
    blockPoint.InsertData(dataPoint);
    blockPolygon.InsertData(dataPolygon);
    blockString.InsertData(dataString);

    ASSERT_EQ(blockInt.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockLong.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockFloat.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockDouble.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockPoint.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockPolygon.EmptyBlockSpace(), 0);
    ASSERT_EQ(blockString.EmptyBlockSpace(), 0);
}

TEST(BlockTests, BlockStatistics)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();
	auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock();
	auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock();
	auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock();
	auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock();
	auto& blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock();
	auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock();

	std::vector<int32_t> dataInt;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloat;
	std::vector<double> dataDouble;
	std::vector<ColmnarDB::Types::Point> dataPoint;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
	std::vector<std::string> dataString;

	for (int i = 0; i < 2; i++)
	{
		dataInt.push_back(1);
		dataLong.push_back(100000000);
		dataFloat.push_back((float) 0.1111);
		dataDouble.push_back((double) 0.1111);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
	}

	dataString.push_back("abc");
	dataString.push_back("aaaaa");
	dataString.push_back("aa");
	dataString.push_back("jkjlo");

	blockInt.InsertData(dataInt);
	blockLong.InsertData(dataLong);
	blockFloat.InsertData(dataFloat);
	blockDouble.InsertData(dataDouble);
	blockPoint.InsertData(dataPoint);
	blockPolygon.InsertData(dataPolygon);
	blockString.InsertData(dataString);

	ASSERT_EQ(blockInt.GetMin(), 1);
	ASSERT_EQ(blockInt.GetMax(), 1);
	ASSERT_EQ(blockInt.GetSum(), 2);
	ASSERT_EQ(blockInt.GetAvg(), 1);

	ASSERT_EQ(blockLong.GetMin(), 100000000);
	ASSERT_EQ(blockLong.GetMax(), 100000000);
	ASSERT_EQ(blockLong.GetSum(), 200000000);
	ASSERT_FLOAT_EQ(blockLong.GetAvg(), 100000000);

	ASSERT_FLOAT_EQ(blockFloat.GetMin(), 0.1111);
	ASSERT_FLOAT_EQ(blockFloat.GetMax(), 0.1111);
	ASSERT_FLOAT_EQ(blockFloat.GetSum(), 0.2222);
	ASSERT_FLOAT_EQ(blockFloat.GetAvg(), 0.1111);

	ASSERT_DOUBLE_EQ(blockDouble.GetMin(), 0.1111);
	ASSERT_DOUBLE_EQ(blockDouble.GetMax(), 0.1111);
	ASSERT_DOUBLE_EQ(blockDouble.GetSum(), 0.2222);
	ASSERT_FLOAT_EQ(blockDouble.GetAvg(), 0.1111);

	ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.GetMin()), "POINT(0 0)");
	ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.GetMax()), "POINT(0 0)");
	ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.GetSum()), "POINT(0 0)");
	ASSERT_FLOAT_EQ(blockPoint.GetAvg(), 0);

	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.GetMin()), "POLYGON((0 0), (0 0))");
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.GetMax()), "POLYGON((0 0), (0 0))");
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.GetSum()), "POLYGON((0 0), (0 0))");
	ASSERT_FLOAT_EQ(blockPolygon.GetAvg(), 0);

	ASSERT_EQ(blockString.GetMin(), "aa");
	ASSERT_EQ(blockString.GetMax(), "jkjlo");
	ASSERT_EQ(blockString.GetSum(), "");
	ASSERT_FLOAT_EQ(blockString.GetAvg(), 0);

	std::vector<int32_t> dataInt2;
	std::vector<int64_t> dataLong2;
	std::vector<float> dataFloat2;
	std::vector<double> dataDouble2;
	std::vector<ColmnarDB::Types::Point> dataPoint2;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon2;
	std::vector<std::string> dataString2;

	for (int i = 0; i < 2; i++)
	{
		dataInt2.push_back(3);
		dataLong2.push_back(300000000);
		dataFloat2.push_back((float) 0.3333);
		dataDouble2.push_back((double) 0.3333);
		dataPoint2.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon2.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
	}

	dataString2.push_back("z");
	dataString2.push_back("zzz");
	dataString2.push_back("dfg");
	dataString2.push_back("wert");

	blockInt.InsertData(dataInt2);
	blockLong.InsertData(dataLong2);
	blockFloat.InsertData(dataFloat2);
	blockDouble.InsertData(dataDouble2);
	blockPoint.InsertData(dataPoint2);
	blockPolygon.InsertData(dataPolygon2);
	blockString.InsertData(dataString2);

	ASSERT_EQ(blockInt.GetMin(), 1);
	ASSERT_EQ(blockInt.GetMax(), 3);
	ASSERT_EQ(blockInt.GetSum(), 8);
	ASSERT_EQ(blockInt.GetAvg(), 2);

	ASSERT_EQ(blockLong.GetMin(), 100000000);
	ASSERT_EQ(blockLong.GetMax(), 300000000);
	ASSERT_EQ(blockLong.GetSum(), 800000000);
	ASSERT_FLOAT_EQ(blockLong.GetAvg(), 200000000);

	ASSERT_FLOAT_EQ(blockFloat.GetMin(), 0.1111);
	ASSERT_FLOAT_EQ(blockFloat.GetMax(), 0.3333);
	ASSERT_FLOAT_EQ(blockFloat.GetSum(), 0.8888);
	ASSERT_FLOAT_EQ(blockFloat.GetAvg(), 0.2222);

	ASSERT_DOUBLE_EQ(blockDouble.GetMin(), 0.1111);
	ASSERT_DOUBLE_EQ(blockDouble.GetMax(), 0.3333);
	ASSERT_DOUBLE_EQ(blockDouble.GetSum(), 0.8888);
	ASSERT_FLOAT_EQ(blockDouble.GetAvg(), 0.2222);

	ASSERT_EQ(blockString.GetMin(), "aa");
	ASSERT_EQ(blockString.GetMax(), "zzz");
	ASSERT_EQ(blockString.GetSum(), "");
	ASSERT_FLOAT_EQ(blockString.GetAvg(), 0);
}