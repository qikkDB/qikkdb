#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/CSVDataImporter.h"
#include "../dropdbase/DataType.h"
#include "../dropdbase/ColumnBase.h"

TEST(CSVDataImportTests, CreateTable)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(true, database->GetTables().find("valid_header") != database->GetTables().end());
}

TEST(CSVDataImportTests, ImportHeader)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);
	
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("longitude") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("latitude") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("targetId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("genderId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("ageId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("wealthIndexId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("hwOsId") != database->GetTables().find("valid_header")->second.GetColumns().end());
}

TEST(CSVDataImportTests, ImportWithoutHeader)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_no_header.csv", false, ',');
	importer.ImportTables(database);

	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C0") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C1") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C2") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C3") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C4") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C5") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C6") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
}

TEST(CSVDataImportTests, GuessTypes)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header")->second.GetColumns().find("longitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header")->second.GetColumns().find("latitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("targetId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("genderId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("ageId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("wealthIndexId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("hwOsId")->second->GetColumnType());
}

TEST(CSVDataImportTests, GuessTypesMessedTypes)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_header_messed_types.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("longitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("latitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_LONG, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("targetId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("genderId")->second->GetColumnType());	
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("ageId")->second->GetColumnType());
}

TEST(CSVDataImportTests, Import)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);

	

	ASSERT_EQ(101, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetData().size());
	ASSERT_EQ(11, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetData().at(10));
	ASSERT_EQ(21.2282657634477f, dynamic_cast<ColumnBase<float>*>(database->GetTables().find("valid_header")->second.GetColumns().at("longitude").get())->GetBlocksList().front()->GetData().at(11));
	ASSERT_EQ(-1, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("genderId").get())->GetBlocksList().front()->GetData().at(12));
	ASSERT_EQ(3, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("hwOsId").get())->GetBlocksList().front()->GetData().at(100));	
}

TEST(CSVDataImportTests, ImportSkipRow)
{
	auto& database = std::make_shared<Database>("testDatabase", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("../../csv_tests/invalid_row_header.csv", true, ',');
	importer.ImportTables(database);



	ASSERT_EQ(100, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("invalid_row_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetData().size());	
}

