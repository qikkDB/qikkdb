#include "gtest/gtest.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/DataType.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"

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
	table.CreateColumn("ColumnBool", COLUMN_BOOL);

	auto columnTypeInt = table.GetColumns().at("ColumnInt")->GetColumnType();
	auto columnTypeLong = table.GetColumns().at("ColumnLong")->GetColumnType();
	auto columnTypeFloat = table.GetColumns().at("ColumnFloat")->GetColumnType();
	auto columnTypeDouble = table.GetColumns().at("ColumnDouble")->GetColumnType();
	auto columnTypePoint = table.GetColumns().at("ColumnPoint")->GetColumnType();
	auto columnTypePolygon = table.GetColumns().at("ColumnPolygon")->GetColumnType();
	auto columnTypeString = table.GetColumns().at("ColumnString")->GetColumnType();
	auto columnTypeBool = table.GetColumns().at("ColumnBool")->GetColumnType();
	
	ASSERT_EQ(columnTypeInt, COLUMN_INT);
	ASSERT_EQ(columnTypeLong, COLUMN_LONG);
	ASSERT_EQ(columnTypeFloat, COLUMN_FLOAT);
	ASSERT_EQ(columnTypeDouble, COLUMN_DOUBLE);
	ASSERT_EQ(columnTypePoint, COLUMN_POINT);
	ASSERT_EQ(columnTypePolygon, COLUMN_POLYGON);
	ASSERT_EQ(columnTypeString, COLUMN_STRING);
	ASSERT_EQ(columnTypeBool, COLUMN_BOOL);
}

TEST(TableTests, InsertData)
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
	table.CreateColumn("ColumnBool", COLUMN_BOOL);

	std::unordered_map<std::string, std::any> data;

	std::vector<int32_t> dataInt({1024});
	std::vector<int64_t> dataLong({1000000000000000000});
	std::vector<float> dataFloat({(float) 0.1111});
	std::vector<double> dataDouble({0.1111111});
	std::vector<ColmnarDB::Types::Point> dataPoint({ PointFactory::FromWkt("POINT(10.11 11.1)")});
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon({ ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") });
	std::vector<std::string> dataString({"randomString"});
	std::vector<bool> dataBool({1});

	data.insert({"ColumnInt",dataInt});
	data.insert({"ColumnLong",dataLong});
	data.insert({"ColumnFloat",dataFloat});
	data.insert({"ColumnDouble",dataDouble});
	data.insert({"ColumnPoint",dataPoint});
	data.insert({"ColumnPolygon",dataPolygon});
	data.insert({"ColumnString",dataString});

	table.InsertData(data);

	auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList();
	ASSERT_EQ(blockInt.front()->GetData()[0], 1024);

	auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList();
	ASSERT_EQ(blockLong.front()->GetData()[0], 1000000000000000000);

	auto& blockFloat = dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList();
	ASSERT_EQ(blockFloat.front()->GetData()[0], (float) 0.1111);

	auto& blockDouble = dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList();
	ASSERT_EQ(blockDouble.front()->GetData()[0], 0.1111111);

	auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList();
	ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.front()->GetData()[0]),"POINT(10.11 11.1)");

	auto& blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList();
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.front()->GetData()[0]), "POLYGON((10 11,11.11 12.13,10 11),(21 30,35.55 36,30.11 20.26,21 30),(61 80.11,90 89.15,112.12 110,61 80.11))");

	auto& blockString = dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList();
	ASSERT_EQ(blockString.front()->GetData()[0], "randomString");
}
