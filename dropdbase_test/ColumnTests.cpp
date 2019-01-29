#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"

TEST(ColumnTests, AddBlock)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);
	table.CreateColumn("ColumnBool", COLUMN_BOOL);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");
	auto& columnBool = table.GetColumns().at("ColumnBool");

	std::vector<int32_t> dataInt({ 1024 });
	std::vector<int64_t> dataLong({ 1000000000000000000 });
	std::vector<float> dataFloat({ (float) 0.1111 });
	std::vector<double> dataDouble({ 0.1111111 });
	std::vector<ColmnarDB::Types::Point> dataPoint({ PointFactory::FromWkt("POINT(10.11 11.1)") });
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon({ ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") });
	std::vector<std::string> dataString({ "randomString" });
	std::vector<bool> dataBool({ 1 });

	auto blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	/*auto blockLong = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockFloat = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockDouble = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockPoint = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockPolygon = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockString = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto blockBool = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);*/
	
	ASSERT_EQ(blockInt.GetData()[0], dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList().front()->GetData()[0]);
}