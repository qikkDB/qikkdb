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
    const std::string path = Configuration::GetInstance().GetDatabaseDir();
    const std::string dbName = "TestDatabase";
    const int32_t blockNum = 2; // number of blocks
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> database;
    std::shared_ptr<Database> renameDatabase;
    virtual void SetUp()
    {
        database = std::make_shared<Database>(dbName.c_str(), blockSize);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());

        // clear directory to make sure, there are no old database files, but do not remove directory:
        boost::filesystem::path path_to_remove(path);
        for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove); it != end_dir_it; ++it)
        {
            boost::filesystem::remove_all(it->path());
        }
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
///  - DropColumn()
///  - DropTable()
///  - DropDatabase()
TEST_F(DatabaseTests, IntegrationTest)
{
    Database::AddToInMemoryDatabaseList(database);

    // create first table with initialized columns:
    std::unordered_map<std::string, DataType> columnsTable1;
    columnsTable1.insert({"colInteger", COLUMN_INT});
    columnsTable1.insert({"colDouble", COLUMN_DOUBLE});
    columnsTable1.insert({"colString", COLUMN_STRING});
    database->CreateTable(columnsTable1, "TestTable1");

    // create second table with initialized columns:
    std::unordered_map<std::string, DataType> columnsTable2;
    std::unordered_map<std::string, bool> columnsTable2Nullabilities;
    columnsTable2.insert({"colInteger", COLUMN_INT});
    columnsTable2.insert({"colDouble", COLUMN_DOUBLE});
    columnsTable2.insert({"colString", COLUMN_STRING});
    columnsTable2.insert({"colFloat", COLUMN_FLOAT});
    columnsTable2.insert({"colLong", COLUMN_LONG});
    columnsTable2.insert({"colPolygon", COLUMN_POLYGON});
    columnsTable2.insert({"colPoint", COLUMN_POINT});
    columnsTable2.insert({"colBool", COLUMN_INT8_T});
    columnsTable2Nullabilities.insert({"colInteger", true});
    columnsTable2Nullabilities.insert({"colDouble", false});
    columnsTable2Nullabilities.insert({"colString", false});
    columnsTable2Nullabilities.insert({"colFloat", false});
    columnsTable2Nullabilities.insert({"colLong", false});
    columnsTable2Nullabilities.insert({"colPolygon", false});
    columnsTable2Nullabilities.insert({"colPoint", false});
    columnsTable2Nullabilities.insert({"colBool", false});
    database->CreateTable(columnsTable2, "TestTable2", columnsTable2Nullabilities);

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
        // insert data to the first table:
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
        dataString1.push_back("QikkDB");
        dataString1.push_back("FastestDBinTheWorld");
        dataString1.push_back("Speed is my second name");
        dynamic_cast<ColumnBase<std::string>*>(colString1.get())->AddBlock(dataString1);

        // insert data to the second table:
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
        dataString2.push_back("Drop database_");
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
        dataPolygon2.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 "
            "80.11,90 89.15,112.12 110, 61 80.11))"));
        dataPolygon2.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
            "87.11,90 89.15,112.12 110, 61 87.11))"));
        dataPolygon2.push_back(ComplexPolygonFactory::FromWkt(
            "POLYGON((15 18, 11.11 12.13, 15 18),(21 38,35.55 36, 30.11 20.26,21 38), (64 80.11,90 "
            "89.15,112.12 110, 64 80.11))"));
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

    for (auto& db : Database::GetDatabaseNames())
    {
        Database::RemoveFromInMemoryDatabaseList(db.c_str());
    }

    // load different database_, but with the same data:
    Database::LoadDatabasesFromDisk();

    auto& loadedTables = Database::GetDatabaseByName(dbName)->GetTables();
    auto& firstTableColumns = loadedTables.at("TestTable1").GetColumns();
    auto& secondTableColumns = loadedTables.at("TestTable2").GetColumns();

    // high level stuff:
    ASSERT_EQ(loadedTables.size(), 2);
    ASSERT_EQ(firstTableColumns.size(), 3);
    ASSERT_EQ(secondTableColumns.size(), 8);

    // first table block counts:
    ASSERT_EQ((firstTableColumns.at("colInteger").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((firstTableColumns.at("colDouble").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((firstTableColumns.at("colString").get())->GetBlockCount(), blockNum);

    ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                  ->GetBlocksList()
                  .at(0)
                  ->BlockCapacity(),
              4);
    ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                  ->GetBlocksList()
                  .at(1)
                  ->BlockCapacity(),
              4);
    ASSERT_EQ(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                  ->GetBlocksList()
                  .at(0)
                  ->BlockCapacity(),
              4);
    ASSERT_EQ(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                  ->GetBlocksList()
                  .at(1)
                  ->BlockCapacity(),
              4);
    ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                  ->GetBlocksList()
                  .at(0)
                  ->BlockCapacity(),
              4);
    ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                  ->GetBlocksList()
                  .at(1)
                  ->BlockCapacity(),
              4);

    // first table nullability of columns:
    ASSERT_TRUE((firstTableColumns.at("colInteger").get())->GetIsNullable());
    ASSERT_TRUE((firstTableColumns.at("colDouble").get())->GetIsNullable());
    ASSERT_TRUE((firstTableColumns.at("colString").get())->GetIsNullable());

    // first table colInteger:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        auto block = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                         ->GetBlocksList()
                         .at(i);
        ASSERT_TRUE(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetIsNullable());
        ASSERT_EQ(data[0], 13);
        ASSERT_EQ(data[1], -2);
        ASSERT_EQ(data[2], 1399);

        ASSERT_EQ(block->GetMin(), -2);
        ASSERT_EQ(block->GetMax(), 1399);
        ASSERT_EQ(block->GetSum(), 1410);
        ASSERT_EQ(block->GetAvg(), 470);
    }

    // first table colDouble:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        auto block = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                         ->GetBlocksList()
                         .at(i);
        ASSERT_TRUE(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetIsNullable());
        ASSERT_DOUBLE_EQ(data[0], 45.98924);
        ASSERT_DOUBLE_EQ(data[1], 999.6665);
        ASSERT_DOUBLE_EQ(data[2], 1.787985);

        ASSERT_DOUBLE_EQ(block->GetMin(), 1.787985);
        ASSERT_DOUBLE_EQ(block->GetMax(), 999.6665);
        ASSERT_FLOAT_EQ(block->GetSum(), 1047.44372f);
        ASSERT_FLOAT_EQ(block->GetAvg(), 349.147908f);
    }

    // first table colString:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        auto block = dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                         ->GetBlocksList()
                         .at(i);
        ASSERT_TRUE(dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetIsNullable());
        ASSERT_EQ(data[0], "QikkDB");
        ASSERT_EQ(data[1], "FastestDBinTheWorld");
        ASSERT_EQ(data[2], "Speed is my second name");

        ASSERT_TRUE(block->GetMin() == "FastestDBinTheWorld");
        ASSERT_TRUE(block->GetMax() == "Speed is my second name");
        ASSERT_TRUE(block->GetSum() == "");
        ASSERT_FLOAT_EQ(block->GetAvg(), 0.0f);
    }

    // second table block count:
    ASSERT_EQ((secondTableColumns.at("colInteger").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colDouble").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colString").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colFloat").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colLong").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colPolygon").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colPoint").get())->GetBlockCount(), blockNum);
    ASSERT_EQ((secondTableColumns.at("colBool").get())->GetBlockCount(), blockNum);

    // second table nullability of columns:
    ASSERT_TRUE((secondTableColumns.at("colInteger").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colDouble").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colString").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colFloat").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colLong").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colPolygon").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colPoint").get())->GetIsNullable());
    ASSERT_FALSE((secondTableColumns.at("colBool").get())->GetIsNullable());

    // second table colInteger:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<int32_t>*>(secondTableColumns.at("colInteger").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_TRUE(dynamic_cast<ColumnBase<int32_t>*>(secondTableColumns.at("colInteger").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetIsNullable());
        ASSERT_EQ(data[0], 1893);
        ASSERT_EQ(data[1], -654);
        ASSERT_EQ(data[2], 196);
    }

    // second table colDouble:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<double>*>(secondTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_FALSE(dynamic_cast<ColumnBase<double>*>(secondTableColumns.at("colDouble").get())
                         ->GetBlocksList()
                         .at(i)
                         ->GetIsNullable());
        ASSERT_DOUBLE_EQ(data[0], 65.77924);
        ASSERT_DOUBLE_EQ(data[1], 9789.685);
        ASSERT_DOUBLE_EQ(data[2], 9.797965);
    }

    // second table colString:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<std::string>*>(secondTableColumns.at("colString").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_FALSE(dynamic_cast<ColumnBase<std::string>*>(secondTableColumns.at("colString").get())
                         ->GetBlocksList()
                         .at(i)
                         ->GetIsNullable());
        ASSERT_EQ(data[0], "Drop database_");
        ASSERT_EQ(data[1], "Is this the fastest DB?");
        ASSERT_EQ(data[2], "Speed of electron");
    }

    // second table colFloat:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<float>*>(secondTableColumns.at("colFloat").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_FALSE(dynamic_cast<ColumnBase<float>*>(secondTableColumns.at("colFloat").get())
                         ->GetBlocksList()
                         .at(i)
                         ->GetIsNullable());
        ASSERT_FLOAT_EQ(data[0], 456.2);
        ASSERT_FLOAT_EQ(data[1], 12.45);
        ASSERT_FLOAT_EQ(data[2], 8.965);
    }

    // second table colPolygon:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                        secondTableColumns.at("colPolygon").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_FALSE(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
                         secondTableColumns.at("colPolygon").get())
                         ->GetBlocksList()
                         .at(i)
                         ->GetIsNullable());
        ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[0]),
                  "POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
                  "80.11, 90 89.15, 112.12 110, 61 80.11))");
        ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[1]),
                  "POLYGON((15 11, 11.11 12.13, 15 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 "
                  "87.11, 90 89.15, 112.12 110, 61 87.11))");
        ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(data[2]),
                  "POLYGON((15 18, 11.11 12.13, 15 18), (21 38, 35.55 36, 30.11 20.26, 21 38), (64 "
                  "80.11, 90 89.15, 112.12 110, 64 80.11))");
    }

    // second table colPoint:
    for (int i = 0; i < blockNum; i++)
    {
        auto data =
            dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(secondTableColumns.at("colPoint").get())
                ->GetBlocksList()
                .at(i)
                ->GetData();
        ASSERT_FALSE(
            dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(secondTableColumns.at("colPoint").get())
                ->GetBlocksList()
                .at(i)
                ->GetIsNullable());
        ASSERT_EQ(PointFactory::WktFromPoint(data[0]), "POINT(10.11 11.1)");
        ASSERT_EQ(PointFactory::WktFromPoint(data[1]), "POINT(12 11.15)");
        ASSERT_EQ(PointFactory::WktFromPoint(data[2]), "POINT(9 8)");
    }

    // second table colBool:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<int8_t>*>(secondTableColumns.at("colBool").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        ASSERT_FALSE(dynamic_cast<ColumnBase<int8_t>*>(secondTableColumns.at("colBool").get())
                         ->GetBlocksList()
                         .at(i)
                         ->GetIsNullable());
        ASSERT_EQ(data[0], -1);
        ASSERT_EQ(data[1], 0);
        ASSERT_EQ(data[2], 1);
    }

    Database::SaveAllToDisk();

    // drop column colBool:
    std::string filePath = Configuration::GetInstance().GetDatabaseDir() + dbName +
                           Database::SEPARATOR + "TestTable2" + Database::SEPARATOR + "colBool" + Database::COLUMN_DATA_EXTENSION;
    ASSERT_TRUE(boost::filesystem::exists(filePath)); // should exist before deletion
    database->DeleteColumnFromDisk(std::string("TestTable2").c_str(), std::string("colBool").c_str());
    ASSERT_FALSE(boost::filesystem::exists(filePath)); // should not exist after deletion

    // drop table TestTable2:
    database->DeleteTableFromDisk(std::string("TestTable2").c_str());
    bool deleted = true;

    std::string prefix = Configuration::GetInstance().GetDatabaseDir() + dbName +
                         Database::SEPARATOR + "TestTable2" + Database::SEPARATOR;

    for (auto& p : boost::filesystem::directory_iterator(Configuration::GetInstance().GetDatabaseDir()))
    {
        // delete files which starts with prefix of db name and table name:
        if (!p.path().string().compare(0, prefix.size(), prefix))
        {
            deleted = false;
        }
    }
    ASSERT_TRUE(deleted) << "DeleteTableFromDisk";

    // drop database_ TestDatabase:
    database->DeleteDatabaseFromDisk();
    deleted = true;

    prefix = Configuration::GetInstance().GetDatabaseDir() + dbName + Database::SEPARATOR;

    for (auto& p : boost::filesystem::directory_iterator(Configuration::GetInstance().GetDatabaseDir()))
    {
        if (!p.path().string().compare(0, prefix.size(), prefix))
        {
            deleted = false;
        }
    }
    ASSERT_TRUE(deleted) << "DeleteDatabaseFromDisk";
    Database::RemoveFromInMemoryDatabaseList("TestDatabase");
}

TEST_F(DatabaseTests, ChangeTableBlockSize)
{
    // create first table with initialized columns:
    std::unordered_map<std::string, DataType> columnsTable1;
    columnsTable1.insert({"colInteger", COLUMN_INT});
    columnsTable1.insert({"colDouble", COLUMN_DOUBLE});
    columnsTable1.insert({"colString", COLUMN_STRING});
    database->CreateTable(columnsTable1, "TestTable");

	auto& tables = database->GetTables();

    auto& table1 = tables.at("TestTable");
    auto& colInteger1 = table1.GetColumns().at("colInteger");
    auto& colDouble1 = table1.GetColumns().at("colDouble");
    auto& colString1 = table1.GetColumns().at("colString");

	for (int i = 0; i < blockNum; i++)
    {
        // insert data to the first table:
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
        dataString1.push_back("QikkDB");
        dataString1.push_back("FastestDBinTheWorld");
        dataString1.push_back("Speed is my second name");
        dynamic_cast<ColumnBase<std::string>*>(colString1.get())->AddBlock(dataString1);
    }

	database->ChangeTableBlockSize("TestTable", 2);

	auto& loadedTables = database->GetTables();
    auto& firstTableColumns = loadedTables.at("TestTable").GetColumns();

	// first table block counts:
    ASSERT_EQ((firstTableColumns.at("colInteger").get())->GetBlockCount(), 3);
    ASSERT_EQ((firstTableColumns.at("colDouble").get())->GetBlockCount(), 3);
    ASSERT_EQ((firstTableColumns.at("colString").get())->GetBlockCount(), 3);

	// first table nullability of columns:
    ASSERT_TRUE((firstTableColumns.at("colInteger").get())->GetIsNullable());
    ASSERT_TRUE((firstTableColumns.at("colDouble").get())->GetIsNullable());
    ASSERT_TRUE((firstTableColumns.at("colString").get())->GetIsNullable());

	// first table colInteger, first block:
    auto data = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(0)
                    ->GetData();
    auto block = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                        ->GetBlocksList()
                        .at(0);
    ASSERT_TRUE(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(0)
                    ->GetIsNullable());
    ASSERT_EQ(data[0], 13);
    ASSERT_EQ(data[1], -2);

    ASSERT_EQ(block->GetMin(), -2);
    ASSERT_EQ(block->GetMax(), 13);
    ASSERT_EQ(block->GetSum(), 11);
    ASSERT_EQ(block->GetAvg(), 5.50);

	// first table colInteger, second block:
    data = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(1)
                    ->GetData();
    block = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                     ->GetBlocksList()
                     .at(1);
    ASSERT_TRUE(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(1)
                    ->GetIsNullable());
    ASSERT_EQ(data[0], 1399);
    ASSERT_EQ(data[1], 13);

    ASSERT_EQ(block->GetMin(), 13);
    ASSERT_EQ(block->GetMax(), 1399);
    ASSERT_EQ(block->GetSum(), 1412);
    ASSERT_EQ(block->GetAvg(), 706);

	// first table colInteger, third block:
    data = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(2)
                    ->GetData();
    block = dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                     ->GetBlocksList()
                     .at(2);
    ASSERT_TRUE(dynamic_cast<ColumnBase<int32_t>*>(firstTableColumns.at("colInteger").get())
                    ->GetBlocksList()
                    .at(2)
                    ->GetIsNullable());
    ASSERT_EQ(data[0], -2);
    ASSERT_EQ(data[1], 1399);

    ASSERT_EQ(block->GetMin(), -2);
    ASSERT_EQ(block->GetMax(), 1399);
    ASSERT_EQ(block->GetSum(), 1397);
    ASSERT_EQ(block->GetAvg(), 698.50);

    // first table colDouble, first block:
    auto data2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(0)
                    ->GetData();
    auto block2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(0);
    ASSERT_TRUE(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(0)
                    ->GetIsNullable());
    ASSERT_DOUBLE_EQ(data2[0], 45.98924);
    ASSERT_DOUBLE_EQ(data2[1], 999.6665);
    
    ASSERT_DOUBLE_EQ(block2->GetMin(), 45.98924);
    ASSERT_DOUBLE_EQ(block2->GetMax(), 999.6665);
    ASSERT_FLOAT_EQ(block2->GetSum(), 1045.6558f);
    ASSERT_FLOAT_EQ(block2->GetAvg(), 522.82787f);

    // first table colDouble, second block:
    data2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(1)
                    ->GetData();
    block2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(1);
    ASSERT_TRUE(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(1)
                    ->GetIsNullable());
    ASSERT_DOUBLE_EQ(data2[0], 1.787985);
    ASSERT_DOUBLE_EQ(data2[1], 45.98924);

    ASSERT_DOUBLE_EQ(block2->GetMin(), 1.787985);
    ASSERT_DOUBLE_EQ(block2->GetMax(), 45.98924);
    ASSERT_FLOAT_EQ(block2->GetSum(), 47.777225f);
    ASSERT_FLOAT_EQ(block2->GetAvg(), 23.8886125f);

    // first table colDouble, third block:
    data2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(2)
                    ->GetData();
    block2 = dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                        ->GetBlocksList()
                        .at(2);
    ASSERT_TRUE(dynamic_cast<ColumnBase<double>*>(firstTableColumns.at("colDouble").get())
                    ->GetBlocksList()
                    .at(2)
                    ->GetIsNullable());
    ASSERT_DOUBLE_EQ(data2[0], 999.6665);
    ASSERT_DOUBLE_EQ(data2[1], 1.787985);

    ASSERT_DOUBLE_EQ(block2->GetMin(), 1.787985);
    ASSERT_DOUBLE_EQ(block2->GetMax(), 999.6665);
    ASSERT_FLOAT_EQ(block2->GetSum(), 1001.454485f);
    ASSERT_FLOAT_EQ(block2->GetAvg(), 500.7272425f);
    
	/*
    // first table colString:
    for (int i = 0; i < blockNum; i++)
    {
        auto data = dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetData();
        auto block = dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                         ->GetBlocksList()
                         .at(i);
        ASSERT_TRUE(dynamic_cast<ColumnBase<std::string>*>(firstTableColumns.at("colString").get())
                        ->GetBlocksList()
                        .at(i)
                        ->GetIsNullable());
        ASSERT_EQ(data[0], "QikkDB");
        ASSERT_EQ(data[1], "FastestDBinTheWorld");
        ASSERT_EQ(data[2], "Speed is my second name");

        ASSERT_TRUE(block->GetMin() == "QikkDB");
        ASSERT_TRUE(block->GetMax() == "Speed is my second name");
        ASSERT_TRUE(block->GetSum() == "");
        ASSERT_FLOAT_EQ(block->GetAvg(), 0.0f);
    }

	*/

	//TODO pridat aj test resizovania celej DB

	//TODO pridaj dalsiu tabulku a opat resizni celu DB
}