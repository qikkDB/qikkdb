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
	Database::RemoveFromInMemoryDatabaseList("TestDbOrderByNULL");

	int32_t blockSize = 1 << 5;
	
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDbOrderByNULL"));
	Database::AddToInMemoryDatabaseList(database);

	std::unordered_map<std::string, DataType> columns;
	columns.emplace("Col1", COLUMN_INT);
	database->CreateTable(columns, "TestTable");

	std::vector<int32_t> expectedResults;
	std::vector<int8_t> expectedNullResults;

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

		if(i < blockSize / 2)
		{
			expectedNullResults.push_back(1);
		}
		else
		{
			expectedNullResults.push_back(0);
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
	for(int32_t i = 0; i < expectedResults.size(); i++){
		int8_t nullBit = (nullBitMask[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
		if(!nullBit)
		{
			ASSERT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
		}
	}

	Database::RemoveFromInMemoryDatabaseList("TestDbOrderByNULL");
}

TEST(DispatcherNullTests, JoinNullTestJoinOnNotNullTables)
{
	srand(42);
	Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL");

	int32_t blockSize = 1 << 5;

	std::shared_ptr<Database> database(std::make_shared<Database>("TestDbJoinNULL"));
	Database::AddToInMemoryDatabaseList(database);

	std::unordered_map<std::string, DataType> columnsR;
	columnsR.emplace("ColA", COLUMN_INT);
	columnsR.emplace("ColJoinA", COLUMN_INT);

	std::unordered_map<std::string, DataType> columnsS;
	columnsS.emplace("ColB", COLUMN_INT);
	columnsS.emplace("ColJoinB", COLUMN_INT);

	database->CreateTable(columnsR, "TestTableR");
	database->CreateTable(columnsS, "TestTableS");

	for(int32_t i = 0, j = blockSize - 1; i < blockSize; i++, j--)
	{
		if(i % 2)
		{
			{
				GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA, ColJoinA) VALUES (null,") + std::to_string(i) + std::string(");"));
				parser.parse();
			}
			{
				GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB, ColJoinB) VALUES (") + std::to_string(j) + std::string(",") + std::to_string(j) + std::string(");"));
				parser.parse();
			}
		}
		else
		{
			{
				GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA, ColJoinA) VALUES (") + std::to_string(i) + std::string(",") + std::to_string(i) + std::string(");"));
				parser.parse();
			}
			{
				GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB, ColJoinB) VALUES (null,") + std::to_string(j) + std::string(");"));
				parser.parse();
			}
		}
	}

	GpuSqlCustomParser parser(database, "SELECT TestTableR.ColA, TestTableS.ColB FROM TestTableR JOIN TestTableS ON ColJoinA = ColJoinB;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());


    auto ColA = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableR").GetColumns().at("ColA").get());
	auto ColB = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableS").GetColumns().at("ColB").get());


    auto& payloadA = result->payloads().at("TestTableR.ColA");
	auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA");

	auto& payloadB = result->payloads().at("TestTableS.ColB");
	auto& nullBitMaskB = result->nullbitmasks().at("TestTableS.ColB");

	ASSERT_EQ(payloadA.intpayload().intdata_size(), payloadB.intpayload().intdata_size());
	for(int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
	{
		int8_t nullBitA = (nullBitMaskA[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
		int8_t nullBitB = (nullBitMaskB[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;

		ASSERT_EQ(nullBitA, nullBitB);

		if(!nullBitA)
		{
			ASSERT_EQ(payloadA.intpayload().intdata()[i], payloadB.intpayload().intdata()[i]);
		}
	}


	Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL");
}