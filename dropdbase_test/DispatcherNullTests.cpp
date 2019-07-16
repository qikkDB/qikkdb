#include "gtest/gtest.h"
#include "../dropdbase/DataType.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/GpuSqlParser/ParserExceptions.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"

TEST(DispatcherNullTests, SelectNullWithWhere)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 1 << 5;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("Col1",COLUMN_INT);
	database->CreateTable(columns,"TestTable");
    std::vector<int> expectedResults;
	for(int i = 0; i < 16; i++)
	{
		if(i % 2 == i/8)
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
			parser.parse();
		}
		else
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
			parser.parse();
            expectedResults.push_back(1);
		}
	}
	GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 = 1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(database->
		GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
	ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
	for (int i = 0; i < payload.intpayload().intdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, IsNullWithPattern)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 1 << 5;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("Col1",COLUMN_INT);
	database->CreateTable(columns,"TestTable");
    std::vector<int8_t> expectedMask;
    int8_t bitMaskPart = 0;
    int nullCount = 0;
	for(int i = 0; i < 16; i++)
	{
		if(i % 2 == i/8)
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
			parser.parse();
            bitMaskPart |= 1 << nullCount++;
		}
		else
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
			parser.parse();
		}
	}
    expectedMask.push_back(bitMaskPart);
	GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 IS NULL;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(database->
		GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& nullBitMask = result->nullbitmasks().at("TestTable.Col1");
	ASSERT_EQ(nullBitMask.size(), expectedMask.size());
	for (int i = 0; i < nullBitMask.size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedMask[i], nullBitMask[i]);
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, IsNotNullWithPattern)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 1 << 5;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("Col1",COLUMN_INT);
	database->CreateTable(columns,"TestTable");
    std::vector<int> expectedResults;
	for(int i = 0; i < 16; i++)
	{
		if(i % 2 == i / 8)
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
			parser.parse();
		}
		else
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
			parser.parse();
            expectedResults.push_back(1);
		}
	}
	GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 IS NOT NULL;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(database->
		GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
	ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
	for (int i = 0; i < payload.intpayload().intdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, OrderByNullTest)
{
	srand(42);
	Database::RemoveFromInMemoryDatabaseList("TestDb");

	int32_t blockSize = 1 << 5;
	
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
	Database::AddToInMemoryDatabaseList(database);

	std::unordered_map<std::string, DataType> columns;
	columns.emplace("Col1", COLUMN_INT);
	database->CreateTable(columns, "TestTable");

	std::vector<int32_t> expectedResults;

	for(int32_t i = 0; i < blockSize; i++)
	{
		if(i % 2)
		{
			GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
			parser.parse();

			expectedResults.push_back(std::numeric_limits<int32_t>::lowest());
		}
		else
		{
			int32_t val = rand() % 1000;

			GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTable (Col1) VALUES (") + std::to_string(val) + std::string(");"));
			parser.parse();

			expectedResults.push_back(val);
		}
	}

	GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable ORDER BY Col1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
	auto& nullBitMask = result->nullbitmasks().at("TestTable.Col1");

	std::stable_sort(expectedResults.begin(), expectedResults.end());

	// Print null column
	int32_t nullColSize = (payload.intpayload().intdata_size() + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);

	ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
	for(int32_t i = 0, k = 0; i < nullColSize; i++){
		for(int32_t j = 7; j >= 0; j--, k++){
			//((nullBitMask[i] >> j) & 1) ? std::printf("%3d : null\n", k) : std::printf("%3d : %d\n", k, payload.intpayload().intdata()[k]);
			ASSERT_FLOAT_EQ(expectedResults[k], payload.intpayload().intdata()[k]);
		}
	}
	//std::printf("\n");

	Database::RemoveFromInMemoryDatabaseList("TestDb");
}