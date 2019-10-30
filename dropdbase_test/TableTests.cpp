#include "gtest/gtest.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/DataType.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"

TEST(TableTests, CreateColumn)
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

    auto columnTypeInt = table.GetColumns().at("ColumnInt")->GetColumnType();
    auto columnTypeLong = table.GetColumns().at("ColumnLong")->GetColumnType();
    auto columnTypeFloat = table.GetColumns().at("ColumnFloat")->GetColumnType();
    auto columnTypeDouble = table.GetColumns().at("ColumnDouble")->GetColumnType();
    auto columnTypePoint = table.GetColumns().at("ColumnPoint")->GetColumnType();
    auto columnTypePolygon = table.GetColumns().at("ColumnPolygon")->GetColumnType();
    auto columnTypeString = table.GetColumns().at("ColumnString")->GetColumnType();

    ASSERT_EQ(columnTypeInt, COLUMN_INT);
    ASSERT_EQ(columnTypeLong, COLUMN_LONG);
    ASSERT_EQ(columnTypeFloat, COLUMN_FLOAT);
    ASSERT_EQ(columnTypeDouble, COLUMN_DOUBLE);
    ASSERT_EQ(columnTypePoint, COLUMN_POINT);
    ASSERT_EQ(columnTypePolygon, COLUMN_POLYGON);
    ASSERT_EQ(columnTypeString, COLUMN_STRING);
}

TEST(TableTests, InsertDataVector)
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

    std::unordered_map<std::string, std::any> data;

    std::vector<int32_t> dataInt({1024});
    std::vector<int64_t> dataLong({1000000000000000000});
    std::vector<float> dataFloat({0.1111f});
    std::vector<double> dataDouble({0.1111111});
    std::vector<ColmnarDB::Types::Point> dataPoint({PointFactory::FromWkt("POINT(10.11 11.1)")});
    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon({ComplexPolygonFactory::FromWkt(
        "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 "
        "89.15,112.12 110, 61 80.11))")});
    std::vector<std::string> dataString({"randomString"});

    data.insert({"ColumnInt", dataInt});
    data.insert({"ColumnInt", dataInt});
    data.insert({"ColumnLong", dataLong});
    data.insert({"ColumnFloat", dataFloat});
    data.insert({"ColumnDouble", dataDouble});
    data.insert({"ColumnPoint", dataPoint});
    data.insert({"ColumnPolygon", dataPolygon});
    data.insert({"ColumnString", dataString});

    table.InsertData(data);

    auto& blockInt =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList();
    ASSERT_EQ(blockInt.front()->GetData()[0], 1024);

    auto& blockLong =
        dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList();
    ASSERT_EQ(blockLong.front()->GetData()[0], 1000000000000000000);

    auto& blockFloat =
        dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList();
    ASSERT_EQ(blockFloat.front()->GetData()[0], (float)0.1111);

    auto& blockDouble =
        dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList();
    ASSERT_EQ(blockDouble.front()->GetData()[0], 0.1111111);

    auto& blockPoint =
        dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())
            ->GetBlocksList();
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.front()->GetData()[0]), "POINT(10.11 11.1)");

    auto& blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                             table.GetColumns().at("ColumnPolygon").get())
                             ->GetBlocksList();
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.front()->GetData()[0]),
              "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "80.11, 90 89.15, 112.12 110, 61 80.11))");

    auto& blockString =
        dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList();
    ASSERT_EQ(blockString.front()->GetData()[0], "randomString");
}

TEST(TableTests, ClusteredIndexInsert)
{
    auto database = std::make_shared<Database>("testDatabase", 4);
    Table table(database, "testTable");
    table.SetSortingColumns({"ColumnInt1", "ColumnInt2"});

    table.CreateColumn("ColumnInt1", COLUMN_INT);
    table.CreateColumn("ColumnInt2", COLUMN_INT);

    std::unordered_map<std::string, std::any> data;

    std::vector<int32_t> dataInt1({2, 1, 5, 8, 102, 67, 5, 1, 12});
    std::vector<int32_t> dataInt2({21, 12, 50, 80, 1020, 670, 60, 13, 120});

    data.insert({"ColumnInt1", dataInt1});
    data.insert({"ColumnInt2", dataInt2});

    table.InsertData(data);

    auto& blockInt =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt1").get())->GetBlocksList();
    ASSERT_EQ(blockInt.size(), 4);
    ASSERT_EQ(blockInt[0]->GetData()[0], 1);
    ASSERT_EQ(blockInt[0]->GetData()[1], 1);
    ASSERT_EQ(blockInt[0]->GetData()[2], 2);
    ASSERT_EQ(blockInt[1]->GetData()[0], 5);
    ASSERT_EQ(blockInt[1]->GetData()[1], 5);
    ASSERT_EQ(blockInt[2]->GetData()[0], 8);
    ASSERT_EQ(blockInt[2]->GetData()[1], 12);
    ASSERT_EQ(blockInt[3]->GetData()[0], 67);
    ASSERT_EQ(blockInt[3]->GetData()[1], 102);

    auto& blockInt2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt2").get())->GetBlocksList();
    ASSERT_EQ(blockInt2.size(), 4);
    ASSERT_EQ(blockInt2[0]->GetData()[0], 12);
    ASSERT_EQ(blockInt2[0]->GetData()[1], 13);
    ASSERT_EQ(blockInt2[0]->GetData()[2], 21);
    ASSERT_EQ(blockInt2[1]->GetData()[0], 50);
    ASSERT_EQ(blockInt2[1]->GetData()[1], 60);
    ASSERT_EQ(blockInt2[2]->GetData()[0], 80);
    ASSERT_EQ(blockInt2[2]->GetData()[1], 120);
    ASSERT_EQ(blockInt2[3]->GetData()[0], 670);
    ASSERT_EQ(blockInt2[3]->GetData()[1], 1020);
}

TEST(TableTests, ClusteredIndexInsertAdvanced)
{
    auto database = std::make_shared<Database>("testDatabase", 256);
    Table table(database, "testTable");
    table.SetSortingColumns({"ColumnInt1", "ColumnInt2", "ColumnInt3", "ColumnInt4", "ColumnInt5",
                             "ColumnInt6", "ColumnInt7"});

    table.CreateColumn("ColumnInt1", COLUMN_INT);
    table.CreateColumn("ColumnInt2", COLUMN_INT);
    table.CreateColumn("ColumnInt3", COLUMN_INT);
    table.CreateColumn("ColumnInt4", COLUMN_INT);
    table.CreateColumn("ColumnInt5", COLUMN_INT);
    table.CreateColumn("ColumnInt6", COLUMN_INT);
    table.CreateColumn("ColumnInt7", COLUMN_INT);

    // size of data 512;
    // sorted vectors
    std::vector<int32_t> sortedDataInt1(
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt2(
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt3(
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt4(
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt5(
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt6(
        {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
         2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
         2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
         1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1,
         1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
         2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
         2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
         1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
         2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
         1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
         2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2});
    std::vector<int32_t> sortedDataInt7(
        {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2,
         2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1,
         1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2,
         2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
         2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1,
         1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
         2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1,
         1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2,
         1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2,
         2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1,
         1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2,
         2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
         2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1,
         1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
         2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1,
         1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2,
         1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2,
         2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});

    // shuffled vectors
    std::unordered_map<std::string, std::any> data;
    std::vector<int32_t> dataInt1(
        {2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1,
         1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1,
         2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
         1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2,
         1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1,
         2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1,
         2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2,
         1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2,
         1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1,
         2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1,
         2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2,
         1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2,
         2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2,
         2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2,
         1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1,
         1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1,
         1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2,
         1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2});
    std::vector<int32_t> dataInt2(
        {1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2,
         1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1,
         1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1,
         1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1,
         1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1,
         2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1,
         1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1,
         1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1,
         1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2,
         1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1,
         2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1,
         2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1,
         1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1,
         1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1,
         2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1,
         1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1,
         2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2});
    std::vector<int32_t> dataInt3(
        {2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1,
         1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2,
         2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1,
         1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1,
         1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2,
         1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2,
         1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1,
         1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1,
         1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2,
         2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2,
         2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2,
         2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2,
         1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1,
         2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2,
         1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2,
         1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2,
         2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1});
    std::vector<int32_t> dataInt4(
        {2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1,
         1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2,
         2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2,
         2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1,
         1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2,
         1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1,
         1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2,
         1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2,
         1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2,
         1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2,
         1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2,
         2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1,
         1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2,
         1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2,
         2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2,
         2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1,
         1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1});
    std::vector<int32_t> dataInt5(
        {2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1,
         1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2,
         2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2,
         2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2,
         2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1,
         1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1,
         1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2,
         2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1,
         1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2,
         1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2,
         2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2,
         1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1,
         2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1,
         2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1,
         1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2,
         1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2,
         2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1,
         2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1});
    std::vector<int32_t> dataInt6(
        {2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2,
         1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2,
         1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2,
         2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1,
         2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2,
         2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2,
         2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1,
         2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1,
         2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
         2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1,
         1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1,
         2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2,
         2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1,
         2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1,
         2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2,
         1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1});
    std::vector<int32_t> dataInt7(
        {2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2,
         1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2,
         2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2,
         2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2,
         1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1,
         2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1,
         1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2,
         1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1,
         1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
         2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1,
         2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1,
         2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2,
         2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2,
         2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1,
         1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2,
         2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
         1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1,
         1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2});

    data.insert({"ColumnInt1", dataInt1});
    data.insert({"ColumnInt2", dataInt2});
    data.insert({"ColumnInt3", dataInt3});
    data.insert({"ColumnInt4", dataInt4});
    data.insert({"ColumnInt5", dataInt5});
    data.insert({"ColumnInt6", dataInt6});
    data.insert({"ColumnInt7", dataInt7});

    table.InsertData(data);

    // First column
    auto& blocksColumnInt1 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt1").get())->GetBlocksList();
    std::vector<int32_t> dataColumn1;
    for (int i = 0; i < blocksColumnInt1.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt1[i]->GetSize(); j++)
        {
            dataColumn1.push_back(blocksColumnInt1[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt1.size(), dataColumn1.size());
    for (int i = 0; i < dataColumn1.size(); i++)
    {
        ASSERT_EQ(sortedDataInt1[i], dataColumn1[i]);
    }

    // Second column
    auto& blocksColumnInt2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt2").get())->GetBlocksList();
    std::vector<int32_t> dataColumn2;
    for (int i = 0; i < blocksColumnInt2.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt2[i]->GetSize(); j++)
        {
            dataColumn2.push_back(blocksColumnInt2[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt2.size(), dataColumn2.size());
    for (int i = 0; i < dataColumn2.size(); i++)
    {
        ASSERT_EQ(sortedDataInt2[i], dataColumn2[i]);
    }

    // Third column
    auto& blocksColumnInt3 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt3").get())->GetBlocksList();
    std::vector<int32_t> dataColumn3;
    for (int i = 0; i < blocksColumnInt3.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt3[i]->GetSize(); j++)
        {
            dataColumn3.push_back(blocksColumnInt3[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt3.size(), dataColumn3.size());
    for (int i = 0; i < dataColumn3.size(); i++)
    {
        ASSERT_EQ(sortedDataInt3[i], dataColumn3[i]);
    }

    // Fourth column
    auto& blocksColumnInt4 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt4").get())->GetBlocksList();
    std::vector<int32_t> dataColumn4;
    for (int i = 0; i < blocksColumnInt4.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt4[i]->GetSize(); j++)
        {
            dataColumn4.push_back(blocksColumnInt4[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt4.size(), dataColumn4.size());
    for (int i = 0; i < dataColumn4.size(); i++)
    {
        ASSERT_EQ(sortedDataInt4[i], dataColumn4[i]);
    }

    // Fifth column
    auto& blocksColumnInt5 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt5").get())->GetBlocksList();
    std::vector<int32_t> dataColumn5;
    for (int i = 0; i < blocksColumnInt5.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt5[i]->GetSize(); j++)
        {
            dataColumn5.push_back(blocksColumnInt5[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt5.size(), dataColumn5.size());
    for (int i = 0; i < dataColumn5.size(); i++)
    {
        ASSERT_EQ(sortedDataInt5[i], dataColumn5[i]);
    }

    // Sixth column
    auto& blocksColumnInt6 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt6").get())->GetBlocksList();
    std::vector<int32_t> dataColumn6;
    for (int i = 0; i < blocksColumnInt6.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt6[i]->GetSize(); j++)
        {
            dataColumn6.push_back(blocksColumnInt6[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt6.size(), dataColumn6.size());
    for (int i = 0; i < dataColumn6.size(); i++)
    {
        ASSERT_EQ(sortedDataInt6[i], dataColumn6[i]);
    }

    // Seventh column
    auto& blocksColumnInt7 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt7").get())->GetBlocksList();
    std::vector<int32_t> dataColumn7;
    for (int i = 0; i < blocksColumnInt7.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt7[i]->GetSize(); j++)
        {
            dataColumn7.push_back(blocksColumnInt7[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt7.size(), dataColumn7.size());
    for (int i = 0; i < dataColumn7.size(); i++)
    {
        ASSERT_EQ(sortedDataInt7[i], dataColumn7[i]);
    }
}

TEST(TableTests, ClusteredIndexInsertWithNullValues_basic)
{
    auto database = std::make_shared<Database>("testDatabase", 4);
    Table table(database, "testTable");
    table.SetSortingColumns({"ColumnInt1", "ColumnInt2"});

    table.CreateColumn("ColumnInt1", COLUMN_INT);
    table.CreateColumn("ColumnInt2", COLUMN_INT);

    std::unordered_map<std::string, std::any> data;
    std::unordered_map<std::string, std::vector<int8_t>> nullMask;

    std::vector<int32_t> dataInt1({2, 1, 5, 8, 102, 67, 5, 1});
    std::vector<int32_t> dataInt2({21, 12, 50, 80, 1020, 670, 60, 13});

    data.insert({"ColumnInt1", dataInt1});
    data.insert({"ColumnInt2", dataInt2});

    std::vector<int8_t> vectorMask1;
    std::vector<int8_t> vectorMask2;
    vectorMask1.push_back(3);
    vectorMask2.push_back(11);

    nullMask.insert({"ColumnInt1", vectorMask1});
    nullMask.insert({"ColumnInt2", vectorMask2});


    table.InsertData(data, false, nullMask);

    auto& blockInt =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt1").get())->GetBlocksList();
    ASSERT_EQ(blockInt.size(), 3);
    ASSERT_EQ(blockInt[0]->GetData()[0], 1);
    ASSERT_EQ(blockInt[0]->GetData()[1], 2);
    ASSERT_EQ(blockInt[0]->GetData()[2], 1);
    ASSERT_EQ(blockInt[1]->GetData()[0], 5);
    ASSERT_EQ(blockInt[1]->GetData()[1], 5);
    ASSERT_EQ(blockInt[1]->GetData()[2], 8);
    ASSERT_EQ(blockInt[2]->GetData()[0], 67);
    ASSERT_EQ(blockInt[2]->GetData()[1], 102);

    auto& blockInt2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt2").get())->GetBlocksList();
    ASSERT_EQ(blockInt2.size(), 3);
    ASSERT_EQ(blockInt2[0]->GetData()[0], 12);
    ASSERT_EQ(blockInt2[0]->GetData()[1], 21);
    ASSERT_EQ(blockInt2[0]->GetData()[2], 13);
    ASSERT_EQ(blockInt2[1]->GetData()[0], 50);
    ASSERT_EQ(blockInt2[1]->GetData()[1], 60);
    ASSERT_EQ(blockInt2[1]->GetData()[2], 80);
    ASSERT_EQ(blockInt2[2]->GetData()[0], 670);
    ASSERT_EQ(blockInt2[2]->GetData()[1], 1020);
}

TEST(TableTests, ClusteredIndexInsertWithNullValues_advanced)
{
    auto database = std::make_shared<Database>("testDatabase", 4);
    Table table(database, "testTable");
    table.SetSortingColumns({"ColumnInt1", "ColumnInt2", "ColumnInt3"});

    table.CreateColumn("ColumnInt1", COLUMN_INT);
    table.CreateColumn("ColumnInt2", COLUMN_INT);
    table.CreateColumn("ColumnInt3", COLUMN_INT);

    std::unordered_map<std::string, std::any> data;
    std::unordered_map<std::string, std::vector<int8_t>> nullMask;

    std::vector<int32_t> dataInt1({9, 5, 5, 7, 12, 4, 8, 5});
    std::vector<int32_t> dataInt2({7, 5, 7, 5, 12, 89, 56, 7});
    std::vector<int32_t> dataInt3({98, 12, 13, 3, 123, 6, 9, 45});

    std::vector<int32_t> sortedDataInt1({7, 12, 5, 4, 5, 5, 8, 9});
    std::vector<int32_t> sortedDataInt2({5, 12, 7, 89, 5, 7, 56, 7});
    std::vector<int32_t> sortedDataInt3({3, 123, 13, 6, 12, 45, 9, 98});

    data.insert({"ColumnInt1", dataInt1});
    data.insert({"ColumnInt2", dataInt2});
    data.insert({"ColumnInt3", dataInt3});

    std::vector<int8_t> vectorMask1;
    std::vector<int8_t> vectorMask2;
    std::vector<int8_t> vectorMask3;
    vectorMask1.push_back(60);
    vectorMask2.push_back(90);
    vectorMask3.push_back(204);

    nullMask.insert({"ColumnInt1", vectorMask1});
    nullMask.insert({"ColumnInt2", vectorMask2});
    nullMask.insert({"ColumnInt3", vectorMask3});

    table.InsertData(data, false, nullMask);

    // First column
    auto& blocksColumnInt1 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt1").get())->GetBlocksList();
    std::vector<int32_t> dataColumn1;
    for (int i = 0; i < blocksColumnInt1.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt1[i]->GetSize(); j++)
        {
            dataColumn1.push_back(blocksColumnInt1[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt1.size(), dataColumn1.size());
    for (int i = 0; i < dataColumn1.size(); i++)
    {
        ASSERT_EQ(sortedDataInt1[i], dataColumn1[i]);
    }

    // Second column
    auto& blocksColumnInt2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt2").get())->GetBlocksList();
    std::vector<int32_t> dataColumn2;
    for (int i = 0; i < blocksColumnInt2.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt2[i]->GetSize(); j++)
        {
            dataColumn2.push_back(blocksColumnInt2[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt2.size(), dataColumn2.size());
    for (int i = 0; i < dataColumn2.size(); i++)
    {
        ASSERT_EQ(sortedDataInt2[i], dataColumn2[i]);
    }

    // Third column
    auto& blocksColumnInt3 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt3").get())->GetBlocksList();
    std::vector<int32_t> dataColumn3;
    for (int i = 0; i < blocksColumnInt3.size(); i++)
    {
        for (int j = 0; j < blocksColumnInt3[i]->GetSize(); j++)
        {
            dataColumn3.push_back(blocksColumnInt3[i]->GetData()[j]);
        }
    }

    ASSERT_EQ(sortedDataInt3.size(), dataColumn3.size());
    for (int i = 0; i < dataColumn3.size(); i++)
    {
        ASSERT_EQ(sortedDataInt3[i], dataColumn3[i]);
    }
}

TEST(TableTests, SavingNecessary)
{
    GpuSqlCustomParser createDatabase(nullptr, "CREATE DATABASE SaveNecDB 10;");
    auto resultPtr = createDatabase.Parse();

    auto database = Database::GetDatabaseByName("SaveNecDB");
    GpuSqlCustomParser parser(database, "CREATE TABLE testTable (ColumnInt int);");
    resultPtr = parser.Parse();

    auto& table = database->GetTables().at("testTable");
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get());

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();

    ASSERT_EQ(false, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());

    GpuSqlCustomParser parser1(database, "ALTER TABLE testTable ADD ColumnInt2 int;");
    resultPtr = parser1.Parse();

    auto& columnInt2 = table.GetColumns().at("ColumnInt2");
    auto castedColumn2 = dynamic_cast<ColumnBase<int32_t>*>(columnInt2.get());

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();

    GpuSqlCustomParser parser2(database, "INSERT INTO testTable (ColumnInt) VALUES (1024);");
    resultPtr = parser2.Parse();
    auto blockInt = castedColumn->GetBlocksList()[0];

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(true, blockInt->GetSaveNecessary());

    GpuSqlCustomParser parserDropDb(database, "DROP DATABASE SaveNecDB;");
    resultPtr = parserDropDb.Parse();
}

TEST(TableTests, SavingNecessaryWithIndex)
{
    GpuSqlCustomParser createDatabase(nullptr, "CREATE DATABASE SaveNecDB 10;");
    auto resultPtr = createDatabase.Parse();

    auto database = Database::GetDatabaseByName("SaveNecDB");
    GpuSqlCustomParser parser(database,
                              "CREATE TABLE testTable (ColumnInt int, INDEX Ind(ColumnInt));");
    resultPtr = parser.Parse();

    auto& table = database->GetTables().at("testTable");
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get());

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();

    ASSERT_EQ(false, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());

    GpuSqlCustomParser parser1(database, "ALTER TABLE testTable ADD ColumnInt2 int;");
    resultPtr = parser1.Parse();

    auto& columnInt2 = table.GetColumns().at("ColumnInt2");
    auto castedColumn2 = dynamic_cast<ColumnBase<int32_t>*>(columnInt2.get());

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();

    GpuSqlCustomParser parser2(database, "INSERT INTO testTable (ColumnInt) VALUES (1024);");
    resultPtr = parser2.Parse();
    auto blockInt = castedColumn->GetBlocksList()[0];

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(true, blockInt->GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();
    blockInt->SetSaveNecessaryToFalse();

    GpuSqlCustomParser parser3(database, "INSERT INTO testTable (ColumnInt2) VALUES (10);");
    resultPtr = parser3.Parse();
    auto blockInt2 = castedColumn2->GetBlocksList()[0];

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(true, blockInt->GetSaveNecessary());
    ASSERT_EQ(true, blockInt2->GetSaveNecessary());

    GpuSqlCustomParser parserDropDb(database, "DROP DATABASE SaveNecDB;");
    resultPtr = parserDropDb.Parse();
}

TEST(TableTests, SavingNecessaryLowLevel)
{
    auto database = std::make_shared<Database>("testDatabase", 10);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get());
    auto& blockInt = castedColumn->AddBlock();

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, blockInt.GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    blockInt.SetSaveNecessaryToFalse();

    ASSERT_EQ(false, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());
    ASSERT_EQ(false, blockInt.GetSaveNecessary());

    table.CreateColumn("ColumnInt2", COLUMN_INT);
    auto& columnInt2 = table.GetColumns().at("ColumnInt2");
    auto castedColumn2 = dynamic_cast<ColumnBase<int32_t>*>(columnInt2.get());
    auto& blockInt2 = castedColumn2->AddBlock();

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(false, blockInt.GetSaveNecessary());
    ASSERT_EQ(true, blockInt2.GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();
    blockInt.SetSaveNecessaryToFalse();
    blockInt2.SetSaveNecessaryToFalse();

    std::unordered_map<std::string, std::any> data;
    std::vector<int32_t> dataInt({1024});
    data.insert({"ColumnInt", dataInt});
    table.InsertData(data);

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(false, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(true, blockInt.GetSaveNecessary());
    ASSERT_EQ(false, blockInt2.GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();
    blockInt.SetSaveNecessaryToFalse();
    blockInt2.SetSaveNecessaryToFalse();

    table.SetSortingColumns({"ColumnInt"});

    std::unordered_map<std::string, std::any> data2;
    std::vector<int32_t> dataInt2({1025});
    data2.insert({"ColumnInt", dataInt2});
    table.InsertData(data2);

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(true, castedColumn->GetSaveNecessary());
    ASSERT_EQ(false, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(true, blockInt.GetSaveNecessary());
    ASSERT_EQ(false, blockInt2.GetSaveNecessary());

    table.SetSaveNecessaryToFalse();
    castedColumn->SetSaveNecessaryToFalse();
    castedColumn2->SetSaveNecessaryToFalse();
    blockInt.SetSaveNecessaryToFalse();
    blockInt2.SetSaveNecessaryToFalse();

    std::unordered_map<std::string, std::any> data3;
    std::vector<int32_t> dataInt3({10});
    data3.insert({"ColumnInt2", dataInt3});
    table.InsertData(data3);

    ASSERT_EQ(true, table.GetSaveNecessary());
    ASSERT_EQ(false, castedColumn->GetSaveNecessary());
    ASSERT_EQ(true, castedColumn2->GetSaveNecessary());
    ASSERT_EQ(false, blockInt.GetSaveNecessary());
    ASSERT_EQ(true, blockInt2.GetSaveNecessary());
}

TEST(TableTests, InsertIntoIsUnique_AllTypes_InsertNullValuesIntoUniqueColumn)
{
    auto database = std::make_shared<Database>("testDatabaseUnique", 10);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    table.CreateColumn("ColumnLong", COLUMN_LONG);
    table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
    table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
    table.CreateColumn("ColumnPoint", COLUMN_POINT);
    table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
    table.CreateColumn("ColumnString", COLUMN_STRING);
    table.CreateColumn("ColumnBool", COLUMN_INT8_T);

    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& columnLong = table.GetColumns().at("ColumnLong");
    auto& columnFloat = table.GetColumns().at("ColumnFloat");
    auto& columnDouble = table.GetColumns().at("ColumnDouble");
    auto& columnPoint = table.GetColumns().at("ColumnPoint");
    auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
    auto& columnString = table.GetColumns().at("ColumnString");
    auto& columnBool = table.GetColumns().at("ColumnBool");

    auto castedColumnInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get());
    auto castedColumnLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get());
    auto castedColumnDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get());
    auto castedColumnFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get());
    auto castedColumnPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get());
    auto castedColumnPolygon =
        dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get());
    auto castedColumnString = dynamic_cast<ColumnBase<std::string>*>(columnString.get());
    auto castedColumnBool = dynamic_cast<ColumnBase<int8_t>*>(columnBool.get());

    castedColumnInt->SetIsUnique(true);
    castedColumnLong->SetIsUnique(true);
    castedColumnDouble->SetIsUnique(true);
    castedColumnFloat->SetIsUnique(true);
    castedColumnPoint->SetIsUnique(true);
    castedColumnPolygon->SetIsUnique(true);
    castedColumnString->SetIsUnique(true);
    castedColumnBool->SetIsUnique(true);

    // Inserting unique values
    std::unordered_map<std::string, std::any> data;

    std::vector<int32_t> dataInt({2, 4, -6});
    std::vector<int64_t> dataLong({489889498840, 489789498848, 282889448871});
    std::vector<double> dataDouble({48.988949, 48.978949, 28.288944});
    std::vector<float> dataFloat({4.8, 2.3, 2.8});

    std::vector<ColmnarDB::Types::Point> dataPoint;
    dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
    dataPoint.push_back(PointFactory::FromWkt("POINT(12 11.15)"));
    dataPoint.push_back(PointFactory::FromWkt("POINT(9 8)"));

    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30), (61 80.11,90 "
        "89.15,112.12 110, 61 80.11))"));
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 87.11,90 "
        "89.15,112.12 110, 61 87.11))"));
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((15 18, 11.11 12.13, 15 18), (21 38,35.55 36, 30.11 "
        "20.26,21 38), (64 80.11,90 89.15,112.12 110, 64 80.11))"));

    std::vector<std::string> dataString;
    dataString.push_back("Hello");
    dataString.push_back("World");
    dataString.push_back("TestString");

    std::vector<int8_t> dataBool;
    dataBool.push_back(-1);
    dataBool.push_back(0);
    dataBool.push_back(1);

    data.insert({"ColumnInt", dataInt});
    data.insert({"ColumnLong", dataLong});
    data.insert({"ColumnFloat", dataFloat});
    data.insert({"ColumnDouble", dataDouble});
    data.insert({"ColumnPoint", dataPoint});
    data.insert({"ColumnPolygon", dataPolygon});
    data.insert({"ColumnString", dataString});
    data.insert({"ColumnBool", dataBool});

    table.InsertData(data);

    auto blockInt =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList();
    auto blockLong =
        dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList();
    auto blockDouble =
        dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList();
    auto blockFloat =
        dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList();
    auto blockPoint =
        dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())
            ->GetBlocksList();
    auto blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                            table.GetColumns().at("ColumnPolygon").get())
                            ->GetBlocksList();
    auto blockString =
        dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList();
    auto blockBool =
        dynamic_cast<ColumnBase<int8_t>*>(table.GetColumns().at("ColumnBool").get())->GetBlocksList();

    ASSERT_EQ(blockInt[0]->GetSize(), 3);
    ASSERT_EQ(blockInt[0]->GetData()[0], 2);
    ASSERT_EQ(blockInt[0]->GetData()[1], 4);
    ASSERT_EQ(blockInt[0]->GetData()[2], -6);

    ASSERT_EQ(blockLong[0]->GetSize(), 3);
    ASSERT_EQ(blockLong[0]->GetData()[0], 489889498840);
    ASSERT_EQ(blockLong[0]->GetData()[1], 489789498848);
    ASSERT_EQ(blockLong[0]->GetData()[2], 282889448871);

    ASSERT_EQ(blockDouble[0]->GetSize(), 3);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[0], 48.988949);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[1], 48.978949);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[2], 28.288944);

    ASSERT_EQ(blockFloat[0]->GetSize(), 3);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[0], 4.8);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[1], 2.3);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[2], 2.8);

    ASSERT_EQ(blockPoint[0]->GetSize(), 3);
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[0]), "POINT(10.11 11.1)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[1]), "POINT(12 11.15)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[2]), "POINT(9 8)");

    ASSERT_EQ(blockPolygon[0]->GetSize(), 3);
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[0]),
              "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "80.11, 90 89.15, 112.12 110, 61 80.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[1]),
              "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "87.11, 90 89.15, 112.12 110, 61 87.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[2]),
              "POLYGON((15 18, 11.11 12.13, 15 18), (21 38, 35.55 36, 30.11 20.26, 21 38), (64 "
              "80.11, 90 89.15, 112.12 110, 64 80.11))");

    ASSERT_EQ(blockString[0]->GetSize(), 3);
    ASSERT_EQ(blockString[0]->GetData()[0], "Hello");
    ASSERT_EQ(blockString[0]->GetData()[1], "World");
    ASSERT_EQ(blockString[0]->GetData()[2], "TestString");

    ASSERT_EQ(blockBool[0]->GetSize(), 3);
    ASSERT_EQ(blockBool[0]->GetData()[0], -1);
    ASSERT_EQ(blockBool[0]->GetData()[1], 0);
    ASSERT_EQ(blockBool[0]->GetData()[2], 1);

    // trying to insert non unique values into isUnique column + trying to insert null values into other isUnique columns which is forbidden
    std::unordered_map<std::string, std::any> data2;
    data2.insert({"ColumnInt", dataInt});
    ASSERT_THROW(table.InsertData(data2), std::length_error);

    std::unordered_map<std::string, std::any> data3;
    data3.insert({"ColumnLong", dataLong});
    ASSERT_THROW(table.InsertData(data3), std::length_error);

    std::unordered_map<std::string, std::any> data4;
    data4.insert({"ColumnFloat", dataFloat});
    ASSERT_THROW(table.InsertData(data4), std::length_error);

    std::unordered_map<std::string, std::any> data5;
    data5.insert({"ColumnDouble", dataDouble});
    ASSERT_THROW(table.InsertData(data5), std::length_error);

    std::unordered_map<std::string, std::any> data6;
    data6.insert({"ColumnPoint", dataPoint});
    ASSERT_THROW(table.InsertData(data6), std::length_error);

    std::unordered_map<std::string, std::any> data7;
    data7.insert({"ColumnPolygon", dataPolygon});
    ASSERT_THROW(table.InsertData(data7), std::length_error);

    std::unordered_map<std::string, std::any> data8;
    data8.insert({"ColumnString", dataString});
    ASSERT_THROW(table.InsertData(data8), std::length_error);

    std::unordered_map<std::string, std::any> data9;
    data9.insert({"ColumnBool", dataBool});
    ASSERT_THROW(table.InsertData(data9), std::length_error);

    ASSERT_EQ(blockInt[0]->GetSize(), 3);
    ASSERT_EQ(blockInt[0]->GetData()[0], 2);
    ASSERT_EQ(blockInt[0]->GetData()[1], 4);
    ASSERT_EQ(blockInt[0]->GetData()[2], -6);

    ASSERT_EQ(blockLong[0]->GetSize(), 3);
    ASSERT_EQ(blockLong[0]->GetData()[0], 489889498840);
    ASSERT_EQ(blockLong[0]->GetData()[1], 489789498848);
    ASSERT_EQ(blockLong[0]->GetData()[2], 282889448871);

    ASSERT_EQ(blockDouble[0]->GetSize(), 3);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[0], 48.988949);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[1], 48.978949);
    ASSERT_DOUBLE_EQ(blockDouble[0]->GetData()[2], 28.288944);

    ASSERT_EQ(blockFloat[0]->GetSize(), 3);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[0], 4.8);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[1], 2.3);
    ASSERT_FLOAT_EQ(blockFloat[0]->GetData()[2], 2.8);

    ASSERT_EQ(blockPoint[0]->GetSize(), 3);
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[0]), "POINT(10.11 11.1)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[1]), "POINT(12 11.15)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint[0]->GetData()[2]), "POINT(9 8)");

    ASSERT_EQ(blockPolygon[0]->GetSize(), 3);
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[0]),
              "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "80.11, 90 89.15, 112.12 110, 61 80.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[1]),
              "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "87.11, 90 89.15, 112.12 110, 61 87.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon[0]->GetData()[2]),
              "POLYGON((15 18, 11.11 12.13, 15 18), (21 38, 35.55 36, 30.11 20.26, 21 38), (64 "
              "80.11, 90 89.15, 112.12 110, 64 80.11))");

    ASSERT_EQ(blockString[0]->GetSize(), 3);
    ASSERT_EQ(blockString[0]->GetData()[0], "Hello");
    ASSERT_EQ(blockString[0]->GetData()[1], "World");
    ASSERT_EQ(blockString[0]->GetData()[2], "TestString");

    ASSERT_EQ(blockBool[0]->GetSize(), 3);
    ASSERT_EQ(blockBool[0]->GetData()[0], -1);
    ASSERT_EQ(blockBool[0]->GetData()[1], 0);
    ASSERT_EQ(blockBool[0]->GetData()[2], 1);
}

TEST(TableTests, InsertIntoIsUnique_AllTypes_InsertDuplicateValuesIntoUniqueColumns)
{
    auto database = std::make_shared<Database>("testDatabaseUnique", 10);
    Table table(database, "testTable");

    table.CreateColumn("ColumnInt", COLUMN_INT);
    table.CreateColumn("ColumnLong", COLUMN_LONG);
    table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
    table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
    table.CreateColumn("ColumnPoint", COLUMN_POINT);
    table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
    table.CreateColumn("ColumnString", COLUMN_STRING);
    table.CreateColumn("ColumnBool", COLUMN_INT8_T);

    auto& columnInt = table.GetColumns().at("ColumnInt");
    auto& columnLong = table.GetColumns().at("ColumnLong");
    auto& columnFloat = table.GetColumns().at("ColumnFloat");
    auto& columnDouble = table.GetColumns().at("ColumnDouble");
    auto& columnPoint = table.GetColumns().at("ColumnPoint");
    auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
    auto& columnString = table.GetColumns().at("ColumnString");
    auto& columnBool = table.GetColumns().at("ColumnBool");

    auto castedColumnInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get());
    auto castedColumnLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get());
    auto castedColumnDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get());
    auto castedColumnFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get());
    auto castedColumnPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get());
    auto castedColumnPolygon =
        dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get());
    auto castedColumnString = dynamic_cast<ColumnBase<std::string>*>(columnString.get());
    auto castedColumnBool = dynamic_cast<ColumnBase<int8_t>*>(columnBool.get());

    castedColumnInt->SetIsUnique(true);
    castedColumnLong->SetIsUnique(true);
    castedColumnDouble->SetIsUnique(true);
    castedColumnFloat->SetIsUnique(true);
    castedColumnPoint->SetIsUnique(true);
    castedColumnPolygon->SetIsUnique(true);
    castedColumnString->SetIsUnique(true);
    castedColumnBool->SetIsUnique(true);

    // Inserting unique values
    std::unordered_map<std::string, std::any> data;

    std::vector<int32_t> dataInt({2, 4, -6});
    std::vector<int64_t> dataLong({489889498840, 489789498848, 282889448871});
    std::vector<double> dataDouble({48.988949, 48.978949, 28.288944});
    std::vector<float> dataFloat({4.8, 2.3, 2.8});

    std::vector<ColmnarDB::Types::Point> dataPoint;
    dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
    dataPoint.push_back(PointFactory::FromWkt("POINT(12 11.15)"));
    dataPoint.push_back(PointFactory::FromWkt("POINT(9 8)"));

    std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30), (61 80.11,90 "
        "89.15,112.12 110, 61 80.11))"));
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 87.11,90 "
        "89.15,112.12 110, 61 87.11))"));
    dataPolygon.push_back(ComplexPolygonFactory::FromWkt(
        "POLYGON((15 18, 11.11 12.13, 15 18), (21 38,35.55 36, 30.11 "
        "20.26,21 38), (64 80.11,90 89.15,112.12 110, 64 80.11))"));

    std::vector<std::string> dataString;
    dataString.push_back("Hello");
    dataString.push_back("World");
    dataString.push_back("TestString");

    std::vector<int8_t> dataBool;
    dataBool.push_back(-1);
    dataBool.push_back(0);
    dataBool.push_back(1);

    data.insert({"ColumnInt", dataInt});
    data.insert({"ColumnLong", dataLong});
    data.insert({"ColumnFloat", dataFloat});
    data.insert({"ColumnDouble", dataDouble});
    data.insert({"ColumnPoint", dataPoint});
    data.insert({"ColumnPolygon", dataPolygon});
    data.insert({"ColumnString", dataString});
    data.insert({"ColumnBool", dataBool});

    table.InsertData(data);

	// trying to insert non unique value into unique column
    castedColumnInt->SetIsUnique(true);
    castedColumnLong->SetIsUnique(false);
    castedColumnDouble->SetIsUnique(false);
    castedColumnFloat->SetIsUnique(false);
    castedColumnPoint->SetIsUnique(false);
    castedColumnPolygon->SetIsUnique(false);
    castedColumnString->SetIsUnique(false);
    castedColumnBool->SetIsUnique(false);

    ASSERT_THROW(table.InsertData(data), std::length_error);

    auto blockInt1 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList();
    auto blockLong1 =
        dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList();
    auto blockDouble1 =
        dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList();
    auto blockFloat1 =
        dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList();
    auto blockPoint1 =
        dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())
            ->GetBlocksList();
    auto blockPolygon1 = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                             table.GetColumns().at("ColumnPolygon").get())
                             ->GetBlocksList();
    auto blockString1 =
        dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList();
    auto blockBool1 =
        dynamic_cast<ColumnBase<int8_t>*>(table.GetColumns().at("ColumnBool").get())->GetBlocksList();

    ASSERT_EQ(blockInt1[0]->GetSize(), 3);
    ASSERT_EQ(blockInt1[0]->GetData()[0], 2);
    ASSERT_EQ(blockInt1[0]->GetData()[1], 4);
    ASSERT_EQ(blockInt1[0]->GetData()[2], -6);

    ASSERT_EQ(blockLong1[0]->GetSize(), 3);
    ASSERT_EQ(blockLong1[0]->GetData()[0], 489889498840);
    ASSERT_EQ(blockLong1[0]->GetData()[1], 489789498848);
    ASSERT_EQ(blockLong1[0]->GetData()[2], 282889448871);

    ASSERT_EQ(blockDouble1[0]->GetSize(), 3);
    ASSERT_DOUBLE_EQ(blockDouble1[0]->GetData()[0], 48.988949);
    ASSERT_DOUBLE_EQ(blockDouble1[0]->GetData()[1], 48.978949);
    ASSERT_DOUBLE_EQ(blockDouble1[0]->GetData()[2], 28.288944);

    ASSERT_EQ(blockFloat1[0]->GetSize(), 3);
    ASSERT_FLOAT_EQ(blockFloat1[0]->GetData()[0], 4.8);
    ASSERT_FLOAT_EQ(blockFloat1[0]->GetData()[1], 2.3);
    ASSERT_FLOAT_EQ(blockFloat1[0]->GetData()[2], 2.8);

    ASSERT_EQ(blockPoint1[0]->GetSize(), 3);
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint1[0]->GetData()[0]), "POINT(10.11 11.1)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint1[0]->GetData()[1]), "POINT(12 11.15)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint1[0]->GetData()[2]), "POINT(9 8)");

    ASSERT_EQ(blockPolygon1[0]->GetSize(), 3);
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon1[0]->GetData()[0]),
              "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "80.11, 90 89.15, 112.12 110, 61 80.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon1[0]->GetData()[1]),
              "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "87.11, 90 89.15, 112.12 110, 61 87.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon1[0]->GetData()[2]),
              "POLYGON((15 18, 11.11 12.13, 15 18), (21 38, 35.55 36, 30.11 20.26, 21 38), (64 "
              "80.11, 90 89.15, 112.12 110, 64 80.11))");

    ASSERT_EQ(blockString1[0]->GetSize(), 3);
    ASSERT_EQ(blockString1[0]->GetData()[0], "Hello");
    ASSERT_EQ(blockString1[0]->GetData()[1], "World");
    ASSERT_EQ(blockString1[0]->GetData()[2], "TestString");

    ASSERT_EQ(blockBool1[0]->GetSize(), 3);
    ASSERT_EQ(blockBool1[0]->GetData()[0], -1);
    ASSERT_EQ(blockBool1[0]->GetData()[1], 0);
    ASSERT_EQ(blockBool1[0]->GetData()[2], 1);

    castedColumnInt->SetIsUnique(false);
    castedColumnLong->SetIsUnique(false);
    castedColumnDouble->SetIsUnique(false);
    castedColumnFloat->SetIsUnique(false);
    castedColumnPoint->SetIsUnique(true);
    castedColumnPolygon->SetIsUnique(false);
    castedColumnString->SetIsUnique(false);
    castedColumnBool->SetIsUnique(false);

    ASSERT_THROW(table.InsertData(data), std::length_error);

    auto blockInt2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList();
    auto blockLong2 =
        dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList();
    auto blockDouble2 =
        dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList();
    auto blockFloat2 =
        dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList();
    auto blockPoint2 =
        dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())
            ->GetBlocksList();
    auto blockPolygon2 = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                             table.GetColumns().at("ColumnPolygon").get())
                             ->GetBlocksList();
    auto blockString2 =
        dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList();
    auto blockBool2 =
        dynamic_cast<ColumnBase<int8_t>*>(table.GetColumns().at("ColumnBool").get())->GetBlocksList();

    ASSERT_EQ(blockInt2[0]->GetSize(), 3);
    ASSERT_EQ(blockInt2[0]->GetData()[0], 2);
    ASSERT_EQ(blockInt2[0]->GetData()[1], 4);
    ASSERT_EQ(blockInt2[0]->GetData()[2], -6);

    ASSERT_EQ(blockLong2[0]->GetSize(), 3);
    ASSERT_EQ(blockLong2[0]->GetData()[0], 489889498840);
    ASSERT_EQ(blockLong2[0]->GetData()[1], 489789498848);
    ASSERT_EQ(blockLong2[0]->GetData()[2], 282889448871);

    ASSERT_EQ(blockDouble2[0]->GetSize(), 3);
    ASSERT_DOUBLE_EQ(blockDouble2[0]->GetData()[0], 48.988949);
    ASSERT_DOUBLE_EQ(blockDouble2[0]->GetData()[1], 48.978949);
    ASSERT_DOUBLE_EQ(blockDouble2[0]->GetData()[2], 28.288944);

    ASSERT_EQ(blockFloat2[0]->GetSize(), 3);
    ASSERT_FLOAT_EQ(blockFloat2[0]->GetData()[0], 4.8);
    ASSERT_FLOAT_EQ(blockFloat2[0]->GetData()[1], 2.3);
    ASSERT_FLOAT_EQ(blockFloat2[0]->GetData()[2], 2.8);

    ASSERT_EQ(blockPoint2[0]->GetSize(), 3);
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint2[0]->GetData()[0]), "POINT(10.11 11.1)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint2[0]->GetData()[1]), "POINT(12 11.15)");
    ASSERT_EQ(PointFactory::WktFromPoint(blockPoint2[0]->GetData()[2]), "POINT(9 8)");

    ASSERT_EQ(blockPolygon2[0]->GetSize(), 3);
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon2[0]->GetData()[0]),
              "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "80.11, 90 89.15, 112.12 110, 61 80.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon2[0]->GetData()[1]),
              "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
              "87.11, 90 89.15, 112.12 110, 61 87.11))");
    ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon2[0]->GetData()[2]),
              "POLYGON((15 18, 11.11 12.13, 15 18), (21 38, 35.55 36, 30.11 20.26, 21 38), (64 "
              "80.11, 90 89.15, 112.12 110, 64 80.11))");

    ASSERT_EQ(blockString2[0]->GetSize(), 3);
    ASSERT_EQ(blockString2[0]->GetData()[0], "Hello");
    ASSERT_EQ(blockString2[0]->GetData()[1], "World");
    ASSERT_EQ(blockString2[0]->GetData()[2], "TestString");

    ASSERT_EQ(blockBool2[0]->GetSize(), 3);
    ASSERT_EQ(blockBool2[0]->GetData()[0], -1);
    ASSERT_EQ(blockBool2[0]->GetData()[1], 0);
    ASSERT_EQ(blockBool2[0]->GetData()[2], 1);
}

TEST(TableTests, InsertIntoIsUnique_Int)
{
	//insert unique values into both columns - one isUnique and one is not
    auto database = std::make_shared<Database>("testDatabaseUnique", 50);
    Table table(database, "testTable");

    table.CreateColumn("ColumnIntA", COLUMN_INT);
    table.CreateColumn("ColumnIntB", COLUMN_INT);

    auto& columnIntA = table.GetColumns().at("ColumnIntA");
    auto& columnIntB = table.GetColumns().at("ColumnIntB");

    auto castedColumnIntA = dynamic_cast<ColumnBase<int32_t>*>(columnIntA.get());
    auto castedColumnIntB = dynamic_cast<ColumnBase<int32_t>*>(columnIntB.get());

    castedColumnIntA->SetIsUnique(true);

    std::unordered_map<std::string, std::any> data;
    std::vector<int32_t> dataIntA({2, 4, -6});
    std::vector<int32_t> dataIntB({21, 41, -61});

    data.insert({"ColumnIntA", dataIntA});
    data.insert({"ColumnIntB", dataIntB});

    table.InsertData(data);

    auto blockIntA =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA[0]->GetSize(), 3);
    ASSERT_EQ(blockIntA[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA[0]->GetData()[2], -6);

    ASSERT_EQ(blockIntB[0]->GetSize(), 3);
    ASSERT_EQ(blockIntB[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB[0]->GetData()[2], -61);

	//trying to insert unique values into isUnique column and non unique values into non unique column
    std::unordered_map<std::string, std::any> data2;
    std::vector<int32_t> dataIntA2({3, 5, -7});
    std::vector<int32_t> dataIntB2({21, 21, -61});

    data2.insert({"ColumnIntA", dataIntA2});
    data2.insert({"ColumnIntB", dataIntB2});

    table.InsertData(data2);

    auto blockIntA2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB2 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA2[0]->GetSize(), 6);
    ASSERT_EQ(blockIntA2[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA2[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA2[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA2[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA2[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA2[0]->GetData()[5], -7);

    ASSERT_EQ(blockIntB2[0]->GetSize(), 6);
    ASSERT_EQ(blockIntB2[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB2[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB2[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB2[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB2[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB2[0]->GetData()[5], -61);

	// trying to insert non unique values into isUnique column and unique values into non unique column
    std::unordered_map<std::string, std::any> data3;
    std::vector<int32_t> dataIntA3({3, 5, -7});
    std::vector<int32_t> dataIntB3({1, 2, 3});

    data3.insert({"ColumnIntA", dataIntA3});
    data3.insert({"ColumnIntB", dataIntB3});

	ASSERT_THROW(table.InsertData(data3), std::length_error);

    auto blockIntA3 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB3 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA3[0]->GetSize(), 6);
    ASSERT_EQ(blockIntA3[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA3[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA3[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA3[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA3[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA3[0]->GetData()[5], -7);
					   
    ASSERT_EQ(blockIntB3[0]->GetSize(), 6);
    ASSERT_EQ(blockIntB3[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB3[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB3[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB3[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB3[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB3[0]->GetData()[5], -61);

	// trying to insert non unique values into isUnique column and non unique values into non unique column
    std::unordered_map<std::string, std::any> data4;
    std::vector<int32_t> dataIntA4({3});
    std::vector<int32_t> dataIntB4({21});

    data4.insert({"ColumnIntA", dataIntA4});
    data4.insert({"ColumnIntB", dataIntB4});

    ASSERT_THROW(table.InsertData(data4), std::length_error);

    auto blockIntA4 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB4 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA4[0]->GetSize(), 6);
    ASSERT_EQ(blockIntA4[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA4[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA4[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA4[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA4[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA4[0]->GetData()[5], -7);
					   
    ASSERT_EQ(blockIntB4[0]->GetSize(), 6);
    ASSERT_EQ(blockIntB4[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB4[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB4[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB4[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB4[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB4[0]->GetData()[5], -61);

	//trynig to insert non unique values - these duplicity values is within one insert
    std::unordered_map<std::string, std::any> data5;
    std::vector<int32_t> dataIntA5({8, 1, 5, 7, 8});
    std::vector<int32_t> dataIntB5({100, 101, 102, 103, 104});

    data5.insert({"ColumnIntA", dataIntA5});
    data5.insert({"ColumnIntB", dataIntB5});

    ASSERT_THROW(table.InsertData(data5), std::length_error);

    auto blockIntA5 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB5 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA5[0]->GetSize(), 6);
    ASSERT_EQ(blockIntA5[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA5[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA5[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA5[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA5[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA5[0]->GetData()[5], -7);
					   
    ASSERT_EQ(blockIntB5[0]->GetSize(), 6);
    ASSERT_EQ(blockIntB5[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB5[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB5[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB5[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB5[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB5[0]->GetData()[5], -61);

	//insert only to non unique column - null value should be inserted in isUnique column but this is forbidden
    std::unordered_map<std::string, std::any> data6;
    std::vector<int32_t> dataIntA6({100, 101});
    std::vector<int32_t> dataIntB6({100, 101});
    data6.insert({"ColumnIntA", dataIntA6});
    data6.insert({"ColumnIntB", dataIntB6});

    std::unordered_map<std::string, std::vector<int8_t>> nullMask;
    std::vector<int8_t> vectorMaskA;
    std::vector<int8_t> vectorMaskB;
    vectorMaskA.push_back(3);
    vectorMaskB.push_back(0);

    nullMask.insert({"ColumnIntA", vectorMaskA});
    nullMask.insert({"ColumnIntB", vectorMaskB});

    ASSERT_THROW(table.InsertData(data6, false, nullMask), std::length_error);

    auto blockIntA6 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB6 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA6[0]->GetSize(), 6);
    ASSERT_EQ(blockIntA6[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA6[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA6[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA6[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA6[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA6[0]->GetData()[5], -7);
					   
    ASSERT_EQ(blockIntB6[0]->GetSize(), 6);
    ASSERT_EQ(blockIntB6[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB6[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB6[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB6[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB6[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB6[0]->GetData()[5], -61);

    ASSERT_EQ(blockIntA6[0]->GetNullBitmask()[0], 0);
    ASSERT_EQ(blockIntB6[0]->GetNullBitmask()[0], 0);

	// insert only to unique column - null value should be inserted in non unique column
    std::unordered_map<std::string, std::any> data7;
    std::vector<int32_t> dataIntA7({1, 8});
    std::vector<int32_t> dataIntB7({1, 8});
    data7.insert({"ColumnIntA", dataIntA7});
    data7.insert({"ColumnIntB", dataIntB7});

    std::unordered_map<std::string, std::vector<int8_t>> nullMask1;
    std::vector<int8_t> vectorMaskA1;
    std::vector<int8_t> vectorMaskB1;
    vectorMaskA1.push_back(0);
    vectorMaskB1.push_back(3);

    nullMask1.insert({"ColumnIntA", vectorMaskA1});
    nullMask1.insert({"ColumnIntB", vectorMaskB1});

    table.InsertData(data7, false, nullMask1);

    auto blockIntA7 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntA").get())->GetBlocksList();
    auto blockIntB7 =
        dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnIntB").get())->GetBlocksList();

    ASSERT_EQ(blockIntA7[0]->GetSize(), 8);
    ASSERT_EQ(blockIntA7[0]->GetData()[0], 2);
    ASSERT_EQ(blockIntA7[0]->GetData()[1], 4);
    ASSERT_EQ(blockIntA7[0]->GetData()[2], -6);
    ASSERT_EQ(blockIntA7[0]->GetData()[3], 3);
    ASSERT_EQ(blockIntA7[0]->GetData()[4], 5);
    ASSERT_EQ(blockIntA7[0]->GetData()[5], -7);
    ASSERT_EQ(blockIntA7[0]->GetData()[6], 1);
    ASSERT_EQ(blockIntA7[0]->GetData()[7], 8);
					   
    ASSERT_EQ(blockIntB7[0]->GetSize(), 8);
    ASSERT_EQ(blockIntB7[0]->GetData()[0], 21);
    ASSERT_EQ(blockIntB7[0]->GetData()[1], 41);
    ASSERT_EQ(blockIntB7[0]->GetData()[2], -61);
    ASSERT_EQ(blockIntB7[0]->GetData()[3], 21);
    ASSERT_EQ(blockIntB7[0]->GetData()[4], 21);
    ASSERT_EQ(blockIntB7[0]->GetData()[5], -61);
    ASSERT_EQ(blockIntB7[0]->GetData()[6], 1);
    ASSERT_EQ(blockIntB7[0]->GetData()[7], 8);

	ASSERT_EQ(blockIntA7[0]->GetNullBitmask()[0], 0);
    ASSERT_EQ(blockIntB7[0]->GetNullBitmask()[0], -128);
}

TEST(TableTests, InsertIntoIsUnique_Int_ThroughConsole)
{
    GpuSqlCustomParser parserCreateDatabase(nullptr, "CREATE DATABASE UniqueDatabase 50;");
    auto resultPtr = parserCreateDatabase.Parse();

    GpuSqlCustomParser parserCreateTable(Database::GetDatabaseByName("UniqueDatabase"),
                                         "CREATE TABLE Table (ColumnIntA INT UNIQUE, ColumnIntB INT);");
    resultPtr = parserCreateTable.Parse();

	GpuSqlCustomParser parser(Database::GetDatabaseByName("UniqueDatabase"),
                              "INSERT INTO Table (ColumnIntA, ColumnIntB) VALUES (1, 2);");
    resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	ASSERT_EQ(result->payloads().at("Table.ColumnIntA").intpayload().intdata_size(), 1);
	ASSERT_EQ(result->payloads().at("Table.ColumnIntB").intpayload().intdata_size(), 1);
    ASSERT_EQ(1, result->payloads().at("Table.ColumnIntA").intpayload().intdata()[0]);
    ASSERT_EQ(2, result->payloads().at("Table.ColumnIntB").intpayload().intdata()[0]);

	GpuSqlCustomParser parserDropDatabase(nullptr, "DROP DATABASE UniqueDatabase;");
    resultPtr = parserDropDatabase.Parse();
}