#include <boost/filesystem.hpp>

#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Configuration.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"

class DatabaseTests : public ::testing::Test
{
protected:
	const std::string path = Configuration::GetInstance().GetDatabaseDir() + "/testDatabase/";
	const std::string dbName = "TestDatabase";
	const int32_t blockNum = 2; //number of blocks
	const int32_t blockSize = 3; //length of a block

	std::shared_ptr<Database> database;
	virtual void SetUp()
	{
		database = std::make_shared<Database>(dbName.c_str(), blockSize);
	}

	virtual void TearDown()
	{
		//clean up occurs when test completes or an exception is thrown
		Database::DestroyDatabase(dbName.c_str());
	}
};


/// Integration test - tests the following fucntions and procedures:
///  - Persist()
///  - SaveAllToDisk()
///  - LoadDatabasesFromDisk()
///  - LoadDatabase()
///  - LoadColumns()
///  - CreateTable()
///  - AddToInMemoryDatabaseList()
TEST_F(DatabaseTests, SaveLoadTest)
{
	boost::filesystem::current_path();

	Database::AddToInMemoryDatabaseList(database);

	//create first table with initialized columns:
	std::unordered_map<std::string, DataType> columnsTable1;
	columnsTable1.insert( {"colInteger", COLUMN_INT} );
	columnsTable1.insert( {"colDouble", COLUMN_DOUBLE} );
	columnsTable1.insert( {"colString", COLUMN_STRING} );
	database->CreateTable(columnsTable1, "TestTable1");

	//create second table with initialized columns:
	std::unordered_map<std::string, DataType> columnsTable2;
	columnsTable2.insert( {"colInteger", COLUMN_INT} );
	columnsTable2.insert( {"colDouble", COLUMN_DOUBLE} );
	columnsTable2.insert( {"colString", COLUMN_STRING} );
	columnsTable2.insert( {"colFloat", COLUMN_FLOAT} );
	columnsTable2.insert( {"colLong", COLUMN_LONG} );
	columnsTable2.insert( {"colPolygon", COLUMN_POLYGON} );
	columnsTable2.insert( {"colPoint", COLUMN_POINT} );
	columnsTable2.insert( {"colBool", COLUMN_INT8_T });
	database->CreateTable(columnsTable2, "TestTable2");

	auto& tables = database->GetTables();

	auto& table1 = tables.at("TestTable1");
	auto& colInteger1 = table1.GetColumns().at("colInteger");
	auto& colDouble1 = table1.GetColumns().at("colDouble");
	auto& colString1 = table1.GetColumns().at("colString");

	auto& table2 = tables.at("TestTable2");
	auto& colInteger2 = table2.GetColumns().at("colInteger");
	auto& colDouble2 = table2.GetColumns().at("colDouble");
	auto& colString2 = table2.GetColumns().at("colString");
	auto& colFloat2 = table2.GetColumns().at("colFloat");
	auto& colLong2 = table2.GetColumns().at("colLong");
	auto& colPolygon2 = table2.GetColumns().at("colPolygon");
	auto& colPoint2 = table2.GetColumns().at("colPoint");
	auto& colBool2 = table2.GetColumns().at("colBool");

	for (int i = 0; i < blockNum; i++)
	{
		//insert data to the first table:
		std::vector<int32_t> dataInteger1;
		dataInteger1.push_back(13);
		dataInteger1.push_back(-2);
		dataInteger1.push_back(1399);
		dynamic_cast<ColumnBase<int32_t>*>(colInteger1.get())->AddBlock(dataInteger1);

		std::vector<double> dataDouble1;
		dataDouble1.push_back(45.98924);
		dataDouble1.push_back(999.6665);
		dataDouble1.push_back(1.787985);
		dynamic_cast<ColumnBase<double>*>(colDouble1.get())->AddBlock(dataDouble1);

		std::vector<std::string> dataString1;
		dataString1.push_back("DropDBase");
		dataString1.push_back("FastestDBinTheWorld");
		dataString1.push_back("Speed is my second name");
		dynamic_cast<ColumnBase<std::string>*>(colString1.get())->AddBlock(dataString1);

		//insert data to the second table:
		std::vector<int32_t> dataInteger2;
		dataInteger2.push_back(1893);
		dataInteger2.push_back(-654);
		dataInteger2.push_back(196);
		dynamic_cast<ColumnBase<int32_t>*>(colInteger2.get())->AddBlock(dataInteger2);

		std::vector<double> dataDouble2;
		dataDouble2.push_back(65.77924);
		dataDouble2.push_back(9789.685);
		dataDouble2.push_back(9.797965);
		dynamic_cast<ColumnBase<double>*>(colDouble2.get())->AddBlock(dataDouble2);

		std::vector<std::string> dataString2;
		dataString2.push_back("Drop database");
		dataString2.push_back("Is this the fastest DB?");
		dataString2.push_back("Speed of electron");
		dynamic_cast<ColumnBase<std::string>*>(colString2.get())->AddBlock(dataString2);

		std::vector<float> dataFloat2;
		dataFloat2.push_back(456.2);
		dataFloat2.push_back(12.45);
		dataFloat2.push_back(8.965);
		dynamic_cast<ColumnBase<float>*>(colFloat2.get())->AddBlock(dataFloat2);

		std::vector<int64_t> dataLong2;
		dataLong2.push_back(489889498840);
		dataLong2.push_back(165648654445);
		dataLong2.push_back(256854586987);
		dynamic_cast<ColumnBase<int64_t>*>(colLong2.get())->AddBlock(dataLong2);

		std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon2;
		dataPolygon2.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
		dataPolygon2.push_back(ComplexPolygonFactory::FromWkt("POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 87.11,90 89.15,112.12 110, 61 87.11))"));
		dataPolygon2.push_back(ComplexPolygonFactory::FromWkt("POLYGON((15 18, 11.11 12.13, 15 18),(21 38,35.55 36, 30.11 20.26,21 38), (64 80.11,90 89.15,112.12 110, 64 80.11))"));
		dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(colPolygon2.get())->AddBlock(dataPolygon2);

		std::vector<ColmnarDB::Types::Point> dataPoint2;
		dataPoint2.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPoint2.push_back(PointFactory::FromWkt("POINT(12 11.15)"));
		dataPoint2.push_back(PointFactory::FromWkt("POINT(9 8)"));
		dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(colPoint2.get())->AddBlock(dataPoint2);

		std::vector<int8_t> dataBool2;
		dataBool2.push_back(-1);
		dataBool2.push_back(0);
		dataBool2.push_back(1);
		dynamic_cast<ColumnBase<int8_t>*>(colBool2.get())->AddBlock(dataBool2);
	}

	std::string storePath = path + dbName;
	boost::filesystem::remove_all(storePath);

	Database::SaveAllToDisk();

	//load different database. but with the same data:
	Database::LoadDatabasesFromDisk();
	
	auto& loadedTables = Database::GetDatabaseByName(dbName)->GetTables();
	auto& firstTableColumns = loadedTables.at("TestTable1").GetColumns();
	auto& secondTableColumns = loadedTables.at("TestTable2").GetColumns();

	//high level stuff:
	ASSERT_EQ(loadedTables.size(), 2);
	ASSERT_EQ(firstTableColumns.size(), 3);
	ASSERT_EQ(secondTableColumns.size(), 8);

	//first table block counts:
	ASSERT_EQ((firstTableColumns.at("colInteger").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((firstTableColumns.at("colDouble").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((firstTableColumns.at("colString").get())->GetBlockCount(), blockNum);

	//first table colInteger:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(data[0], 13);
		ASSERT_EQ(data[1], -2);
		ASSERT_EQ(data[2], 1399);
	}

	//first table colDouble:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())->GetBlocksList().at(i)->GetData();
		ASSERT_DOUBLE_EQ(data[0], 45.98924);
		ASSERT_DOUBLE_EQ(data[1], 999.6665);
		ASSERT_DOUBLE_EQ(data[2], 1.787985);
	}

	//first table colString:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(data[0], "DropDBase");
		ASSERT_EQ(data[1], "FastestDBinTheWorld");
		ASSERT_EQ(data[2], "Speed is my second name");
	}

	//second table block count:
	ASSERT_EQ((secondTableColumns.at("colInteger").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colDouble").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colString").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colFloat").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colDouble").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colPolygon").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colPoint").get())->GetBlockCount(), blockNum);
	ASSERT_EQ((secondTableColumns.at("colBool").get())->GetBlockCount(), blockNum);

	//second table colInteger:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<int32_t>*>(secondTableColumns.at("colInteger").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(data[0], 1893);
		ASSERT_EQ(data[1], -654);
		ASSERT_EQ(data[2], 196);
	}

	//second table colDouble:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<double>*>(secondTableColumns.at("colDouble").get())->GetBlocksList().at(i)->GetData();
		ASSERT_DOUBLE_EQ(data[0], 65.77924);
		ASSERT_DOUBLE_EQ(data[1], 9789.685);
		ASSERT_DOUBLE_EQ(data[2], 9.797965);
	}

	//second table colString:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<std::string>*>(secondTableColumns.at("colString").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(data[0], "Drop database");
		ASSERT_EQ(data[1], "Is this the fastest DB?");
		ASSERT_EQ(data[2], "Speed of electron");
	}

	//second table colFloat:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<float>*>(secondTableColumns.at("colFloat").get())->GetBlocksList().at(i)->GetData();
		ASSERT_FLOAT_EQ(data[0], 456.2);
		ASSERT_FLOAT_EQ(data[1], 12.45);
		ASSERT_FLOAT_EQ(data[2], 8.965);
	}

	//second table colPolygon:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(secondTableColumns.at("colPolygon").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[0]), "POLYGON((10 11,11.11 12.13,10 11),(21 30,35.55 36,30.11 20.26,21 30),(61 80.11,90 89.15,112.12 110,61 80.11))");
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[1]), "POLYGON((15 11,11.11 12.13,15 11),(21 30,35.55 36,30.11 20.26,21 30),(61 87.11,90 89.15,112.12 110,61 87.11))");
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[2]), "POLYGON((15 18,11.11 12.13,15 18),(21 38,35.55 36,30.11 20.26,21 38),(64 80.11,90 89.15,112.12 110,64 80.11))");
	}

	//second table colPoint:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(secondTableColumns.at("colPoint").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(PointFactory::WktFromPoint(data[0]), "POINT(10.11 11.1)");
		ASSERT_EQ(PointFactory::WktFromPoint(data[1]), "POINT(12 11.15)");
		ASSERT_EQ(PointFactory::WktFromPoint(data[2]), "POINT(9 8)");
	}

	//second table colBool:
	for (int i = 0; i < blockNum; i++)
	{
		auto data = dynamic_cast<ColumnBase<int8_t>*>(secondTableColumns.at("colBool").get())->GetBlocksList().at(i)->GetData();
		ASSERT_EQ(data[0], -1);
		ASSERT_EQ(data[1], 0);
		ASSERT_EQ(data[2], 1);
	}
}
