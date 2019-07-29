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

/*
TEST(DispatcherNullTests, JoinIsNotNullTest)
{
	srand(42);
	Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL2");

	int32_t blockSize = 1 << 5;

	std::shared_ptr<Database> database(std::make_shared<Database>("TestDbJoinNULL2"));
	Database::AddToInMemoryDatabaseList(database);

	std::unordered_map<std::string, DataType> columnsR;
	columnsR.emplace("ColA", COLUMN_INT);
	columnsR.emplace("ColJoinA", COLUMN_INT);

	std::unordered_map<std::string, DataType> columnsS;
	columnsS.emplace("ColB", COLUMN_INT);
	columnsS.emplace("ColJoinB", COLUMN_INT);

	database->CreateTable(columnsR, "TestTableR");
	database->CreateTable(columnsS, "TestTableS");

	for (int32_t i = 0, j = blockSize - 1; i < blockSize; i++, j--)
	{
		if (i % 2)
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

	GpuSqlCustomParser parser(database, "SELECT TestTableR.ColA FROM TestTableR JOIN TestTableS ON ColJoinA = ColJoinB WHERE TestTableR.ColA IS NOT NULL;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto ColA = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableR").GetColumns().at("ColA").get());
	auto ColJoinA = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableR").GetColumns().at("ColJoinA").get());
	auto ColJoinB = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableS").GetColumns().at("ColJoinB").get());

	std::vector<int32_t> expectedResults;

	for (int32_t leftBlockIdx = 0; leftBlockIdx < ColJoinA->GetBlockCount(); leftBlockIdx++)
	{
		auto leftRetBlock = ColA->GetBlocksList()[leftBlockIdx];
		auto leftBlock = ColJoinA->GetBlocksList()[leftBlockIdx];
		for (int32_t leftRowIdx = 0; leftRowIdx < leftBlock->GetSize(); leftRowIdx++)
		{
			for (int32_t rightBlockIdx = 0; rightBlockIdx < ColJoinB->GetBlockCount(); rightBlockIdx++)
			{
				auto rightBlock = ColJoinB->GetBlocksList()[rightBlockIdx];
				for (int32_t rightRowIdx = 0; rightRowIdx < rightBlock->GetSize(); rightRowIdx++)
				{
					int8_t nullBit = (leftRetBlock->GetNullBitmask()[leftRowIdx / (sizeof(int8_t) * 8)] >> (leftRowIdx % (sizeof(int8_t) * 8))) & 1;
					if (leftBlock->GetData()[leftRowIdx] == rightBlock->GetData()[rightRowIdx] && nullBit == 0)
					{
						expectedResults.push_back(leftRetBlock->GetData()[leftRowIdx]);
					}
				}
			}
		}
	}

	auto& payloadA = result->payloads().at("TestTableR.ColA");
	auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA");

	ASSERT_EQ(payloadA.intpayload().intdata().size(), expectedResults.size());

	for (int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
	{
		int8_t nullBitA = (nullBitMaskA[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;

		ASSERT_EQ(payloadA.intpayload().intdata()[i], expectedResults[i]);
		ASSERT_EQ(nullBitA, 0);
	}


	Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL2");
}*/

TEST(DispatcherNullTests, JoinNullTestJoinOnNullTables)
{
	srand(42);
	Database::RemoveFromInMemoryDatabaseList("TestDbJoinOnNULL");

	int32_t blockSize = 1 << 5;

	std::shared_ptr<Database> database(std::make_shared<Database>("TestDbJoinOnNULL"));
	Database::AddToInMemoryDatabaseList(database);

	std::unordered_map<std::string, DataType> columnsR;
	columnsR.emplace("ColA", COLUMN_INT);

	std::unordered_map<std::string, DataType> columnsS;
	columnsS.emplace("ColB", COLUMN_INT);

	database->CreateTable(columnsR, "TestTableR");
	database->CreateTable(columnsS, "TestTableS");

	for (int32_t i = 0; i < blockSize; i++)
	{
		if (i % 2)
		{
			GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA) VALUES (null);"));
			parser.parse();
		}
		else
		{
			GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA) VALUES (") + std::to_string(i) + std::string(");"));
			parser.parse();
		}

		if (i < blockSize / 2)
		{
			GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB) VALUES (null);"));
			parser.parse();
		}
		else
		{
			GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB) VALUES (") + std::to_string(i) + std::string(");"));
			parser.parse();
		}
	}

	GpuSqlCustomParser parser(database, "SELECT TestTableR.ColA, TestTableS.ColB FROM TestTableR JOIN TestTableS ON ColA = ColB;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto ColA = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableR").GetColumns().at("ColA").get());
	auto ColB = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("TestTableS").GetColumns().at("ColB").get());

	auto& payloadA = result->payloads().at("TestTableR.ColA");
	auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA");

	auto& payloadB = result->payloads().at("TestTableS.ColB");
	auto& nullBitMaskB = result->nullbitmasks().at("TestTableS.ColB");

	ASSERT_EQ(payloadA.intpayload().intdata_size(), payloadB.intpayload().intdata_size());
	for (int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
	{
		int8_t nullBitA = (nullBitMaskA[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;
		int8_t nullBitB = (nullBitMaskB[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) & 1;

		ASSERT_EQ(nullBitA, nullBitB);

		if (!nullBitA)
		{
			ASSERT_EQ(payloadA.intpayload().intdata()[i], payloadB.intpayload().intdata()[i]);
		}
	}


	Database::RemoveFromInMemoryDatabaseList("TestDbJoinOnNULL");
}

// == GROUP BY ==
TEST(DispatcherNullTests, GroupByNullKeySum)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_INT);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<int32_t, int32_t> expectedResults;
	int32_t expectedValueAtNull = 0;
	for (int i = 0; i < 32; i++)
	{
		bool nullKey = (i % 4 == 2);
		int32_t intKey = i % 4;
		int32_t intVal = 2;
		std::string key = (nullKey ? "NULL" : std::to_string(intKey));
		std::string val = std::to_string(intVal);
		if (nullKey)
		{
			expectedValueAtNull += intVal;
		}
		else
		{
			if (expectedResults.find(intKey) == expectedResults.end())
			{
				expectedResults.insert({ intKey, intVal });
			}
			else
			{
				expectedResults[intKey] += intVal;
			}
		}
		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  NULL    | 16
	//  0       | 16
	//  1       | 16
	//  3       | 16
	for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		ASSERT_FALSE(valIsNull);
		if (keyIsNull)
		{
			ASSERT_EQ(expectedValueAtNull, valuesResult.intpayload().intdata()[i]);
		}
		else
		{
			ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) == expectedResults.end());
			ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]), valuesResult.intpayload().intdata()[i]);
		}
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueSum)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_INT);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<int32_t, int32_t> expectedResults;
	std::unordered_map<int32_t, bool> expectedValueNullMask;
	for (int i = 0; i < 32; i++)
	{
		int32_t intKey = i % 4;
		int32_t intVal = 2;
		bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
		std::string key = std::to_string(intKey);
		std::string val = (nullValue ? "NULL" : std::to_string(intVal));
		if (nullValue)
		{
			if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ intKey, true });
			}
		}
		else
		{
			// "turn of" null
			if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ intKey, false });
			}
			else
			{
				expectedValueNullMask[intKey] = false;
			}
			// aggregate value
			if (expectedResults.find(intKey) == expectedResults.end())
			{
				expectedResults.insert({ intKey, intVal });
			}
			else
			{
				expectedResults[intKey] += intVal;
			}
		}

		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  0       | 16
	//  1       | 16
	//  2       | 8
	//  3       | NULL
	for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		const int32_t key = keysResult.intpayload().intdata()[i];
		// Check nulls
		ASSERT_FALSE(keyIsNull) << " at result row " << i;
		ASSERT_FALSE(expectedValueNullMask.find(keysResult.intpayload().intdata()[i]) == expectedValueNullMask.end())
			<< " bad key at result row " << i;
		ASSERT_EQ(expectedValueNullMask.at(keysResult.intpayload().intdata()[i]), valIsNull) << " at result row " << i;
		if (!valIsNull)
		{
			// Check value
			ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) == expectedResults.end())
				<< " bad key at result row " << i;
			ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]), valuesResult.intpayload().intdata()[i])
				<< " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
		}
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueAvg)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_INT);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<int32_t, int32_t> expectedResults;
	std::unordered_map<int32_t, bool> expectedValueNullMask;
	const int INT_VAL_CONST = 2;
	for (int i = 0; i < 32; i++)
	{
		int32_t intKey = i % 4;
		bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
		std::string key = std::to_string(intKey);
		std::string val = (nullValue ? "NULL" : std::to_string(INT_VAL_CONST));
		if (nullValue)
		{
			if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ intKey, true });
			}
		}
		else
		{
			// "turn of" null
			if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ intKey, false });
			}
			else
			{
				expectedValueNullMask[intKey] = false;
			}
			// set value
			if (expectedResults.find(intKey) == expectedResults.end())
			{
				expectedResults.insert({ intKey, INT_VAL_CONST });
			}
		}

		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, AVG(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("AVG(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("AVG(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("AVG(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  0       | 2
	//  1       | 2
	//  2       | 2
	//  3       | NULL
	for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		const int32_t key = keysResult.intpayload().intdata()[i];
		// Check nulls
		ASSERT_FALSE(keyIsNull) << " at result row " << i;
		ASSERT_FALSE(expectedValueNullMask.find(keysResult.intpayload().intdata()[i]) == expectedValueNullMask.end()) << " bad key at result row " << i;
		ASSERT_EQ(expectedValueNullMask.at(keysResult.intpayload().intdata()[i]), valIsNull) << " at result row " << i;
		if (!valIsNull)
		{
			// Check keys and values
			ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) == expectedResults.end())
				<< " bad key at result row " << i;
			ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]), valuesResult.intpayload().intdata()[i])
				<< " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
		}
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueCount)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_INT);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<int32_t, int64_t> expectedResults;
	for (int i = 0; i < 32; i++)
	{
		int32_t intKey = i % 4;
		int32_t intVal = 2;
		bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
		std::string key = std::to_string(intKey);
		std::string val = (nullValue ? "NULL" : std::to_string(intVal));
		if (nullValue)
		{
			if (expectedResults.find(intKey) == expectedResults.end())
			{
				expectedResults.insert({ intKey, 0 });
			}
		}
		else
		{
			if (expectedResults.find(intKey) == expectedResults.end())
			{
				expectedResults.insert({ intKey, 1 });
			}
			else
			{
				expectedResults[intKey]++;
			}
		}

		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, COUNT(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("COUNT(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("COUNT(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("COUNT(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  0       | 8
	//  1       | 8
	//  2       | 4
	//  3       | 0
	for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		const int32_t key = keysResult.intpayload().intdata()[i];
		// Check nulls (there should be no null value)
		ASSERT_FALSE(keyIsNull) << " at result row " << i;
		ASSERT_FALSE(valIsNull) << " at result row " << i;
		// Check value
		ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) == expectedResults.end()) << " bad key at result row " << i;
		ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]), valuesResult.int64payload().int64data()[i]) << " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}


//== GROUP BY String ==
TEST(DispatcherNullTests, GroupByStringNullKeySum)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_STRING);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<std::string, int32_t> expectedResults;
	int32_t expectedValueAtNull = 0;
	for (int i = 0; i < 32; i++)
	{
		bool nullKey = (i % 4 == 2);
		std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : "XYZ"));
		int32_t intVal = 2;
		std::string key = (nullKey ? "NULL" : ("\"" + strKey + "\""));
		std::string val = std::to_string(intVal);
		if (nullKey)
		{
			expectedValueAtNull += intVal;
		}
		else
		{
			if (expectedResults.find(strKey) == expectedResults.end())
			{
				expectedResults.insert({ strKey, intVal });
			}
			else
			{
				expectedResults[strKey] += intVal;
			}
		}
		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  NULL    | 16
	//  Apple   | 16
	//  Nut     | 16
	//  XYZ     | 16
	ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
	ASSERT_EQ(4, valuesResult.intpayload().intdata_size());
	for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		std::cout << i << ": " << (keyIsNull? "-NULL-" : keysResult.stringpayload().stringdata()[i]) << " | " << valuesResult.intpayload().intdata()[i] << std::endl;
		ASSERT_FALSE(valIsNull);
		if (keyIsNull)
		{
			ASSERT_EQ(expectedValueAtNull, valuesResult.intpayload().intdata()[i]);
		}
		else
		{
			ASSERT_FALSE(expectedResults.find(keysResult.stringpayload().stringdata()[i]) == expectedResults.end()) << keysResult.stringpayload().stringdata()[i];
			ASSERT_EQ(expectedResults.at(keysResult.stringpayload().stringdata()[i]), valuesResult.intpayload().intdata()[i]) << keysResult.stringpayload().stringdata()[i];
		}
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByStringNullValueSum)
{
	Database::RemoveFromInMemoryDatabaseList("TestDb");
	int blockSize = 8;
	std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
	Database::AddToInMemoryDatabaseList(database);
	std::unordered_map<std::string, DataType> columns;
	columns.emplace("colKeys", COLUMN_STRING);
	columns.emplace("colVals", COLUMN_INT);
	database->CreateTable(columns, "TestTable");
	std::unordered_map<std::string, int32_t> expectedResults;
	std::unordered_map<std::string, bool> expectedValueNullMask;
	for (int i = 0; i < 24; i++)
	{
		std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : (i % 4 == 2 ? "Straw" : "car0")));
		int32_t intVal = 2;
		bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
		std::string key = "\"" + strKey + "\"";
		std::string val = (nullValue ? "NULL" : std::to_string(intVal));
		if (nullValue)
		{
			if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ strKey, true });
			}
		}
		else
		{
			// "turn of" null
			if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
			{
				expectedValueNullMask.insert({ strKey, false });
			}
			else
			{
				expectedValueNullMask[strKey] = false;
			}
			// aggregate value
			if (expectedResults.find(strKey) == expectedResults.end())
			{
				expectedResults.insert({ strKey, intVal });
			}
			else
			{
				expectedResults[strKey] += intVal;
			}
		}

		std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");") << std::endl;
		GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");");
		parser.parse();
	}

	GpuSqlCustomParser parser(database, "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
	auto resultPtr = parser.parse();
	auto responseMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
	ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
	const std::string& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys");
	const std::string& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)");
	auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
	auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

	// Result should look like:
	//  colKeys | colVals
	//  Apple   | 16
	//  Nut     | 16
	//  Straw   | 8
	//  car0    | NULL
	ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
	ASSERT_EQ(4, valuesResult.intpayload().intdata_size());
	for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
	{
		const char keyChar = keysNullMaskResult[i / 8];
		const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
		const char valChar = valuesNullMaskResult[i / 8];
		const bool valIsNull = ((valChar >> (i % 8)) & 1);
		const int32_t key = keysResult.intpayload().intdata()[i];
		std::cout << i << ": " << (keyIsNull? "-NULL-" : keysResult.stringpayload().stringdata()[i]) << " | " <<
				(valIsNull? "-NULL-" : std::to_string(valuesResult.intpayload().intdata()[i])) << std::endl;
		// Check nulls
		ASSERT_FALSE(keyIsNull) << " at result row " << i;
		ASSERT_FALSE(expectedValueNullMask.find(keysResult.stringpayload().stringdata()[i]) == expectedValueNullMask.end())
			<< " bad key at result row " << i;
		ASSERT_EQ(expectedValueNullMask.at(keysResult.stringpayload().stringdata()[i]), valIsNull) << " at result row " << i;
		if (!valIsNull)
		{
			// Check value
			ASSERT_FALSE(expectedResults.find(keysResult.stringpayload().stringdata()[i]) == expectedResults.end())
				<< " bad key at result row " << i;
			ASSERT_EQ(expectedResults.at(keysResult.stringpayload().stringdata()[i]), valuesResult.intpayload().intdata()[i])
				<< " with key " << keysResult.stringpayload().stringdata()[i] << " at result row " << i;
		}
	}
	Database::RemoveFromInMemoryDatabaseList("TestDb");
}
