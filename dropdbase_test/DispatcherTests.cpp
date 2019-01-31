#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"

std::vector<std::string> tableNames = { "TableA" };
std::vector<DataType> columnTypes = { {COLUMN_INT},{COLUMN_INT} };
std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 11,0,tableNames,columnTypes);

TEST(DispatcherTests, GtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024) > 5)
			{
				expectedResult.push_back(j % 1024);
			}
	}
	
	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, GtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 500 > colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500 > j % 1024)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, GtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger2 FROM TableA WHERE colInteger2 > colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048) > (j % 1024))
			{
				expectedResult.push_back(j % 2048);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger2");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, GtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 10 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, GtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 > 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, LtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 < 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024) < 5)
			{
				expectedResult.push_back(j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, LtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 500 < colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500 < j % 1024)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}