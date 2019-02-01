#include <cmath>

#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"

std::vector<std::string> tableNames = { "TableA" };
std::vector<DataType> columnTypes = { {COLUMN_INT},{COLUMN_INT},{COLUMN_LONG},{COLUMN_LONG},{COLUMN_FLOAT},{COLUMN_FLOAT},{COLUMN_DOUBLE},{COLUMN_DOUBLE} };
std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 2, 1 << 11, false , tableNames, columnTypes);
/////////////////////
//   ">" operator
/////////////////////

//INT ">"
TEST(DispatcherTests, IntGtColumnConst)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntGtConstColumn)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntGtColumnColumn)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntGtConstConstTrue)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 > 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

// LONG ">"
TEST(DispatcherTests, LongGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 > 500000000;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (static_cast<int64_t>(2 * pow(10, 18)) + j % 1024 > 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 500000000 > colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500000000 > static_cast<int64_t>(2 * pow(10, 18)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong2 FROM TableA WHERE colLong2 > colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((static_cast<int64_t>(2 * pow(10, 18)) + j % 2048) > (static_cast<int64_t>(2 * pow(10, 18)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 2048);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong2");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 10 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 > 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

//FLOAT ">"
TEST(DispatcherTests, FloatGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 > 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) > 5.5)
			{
				expectedResult.push_back((float)(j % 1024 + 0.1111));
			}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5.5 > colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) < 5.5)
			{
				expectedResult.push_back((float)(j % 1024 + 0.1111));
			}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat2 FROM TableA WHERE colFloat2 > colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((float)(j % 2048 + 0.1111)) > ((float)(j % 1024 + 0.1111)))
			{
				expectedResult.push_back((float)(j % 2048 + 0.1111));
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat2");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 10 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((float)(j % 1024 + 0.1111));
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 > 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), 0);
}

//DOUBLE ">"
TEST(DispatcherTests, DoubleGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 > 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) > 5.5)
			{
				expectedResult.push_back(j % 1024 + 0.1111111);
			}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5.5 > colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) < 5.5)
			{
				expectedResult.push_back(j % 1024 + 0.1111111);
			}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble2 FROM TableA WHERE colDouble2 > colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048 + 0.1111111) > (j % 1024 + 0.1111111))
			{
				expectedResult.push_back(j % 2048 + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble2");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 10 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(j % 1024 + 0.1111111);
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 > 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), 0);
}

/////////////////////
//   "<" operator
/////////////////////

//INT "<"
TEST(DispatcherTests, IntLtColumnConst)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntLtConstColumn)
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 < colInteger2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024) < (j % 2048))
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 < 10;");
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

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 10 < 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

// LONG "<"
TEST(DispatcherTests, LongLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 < 500000000;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (2 * (10, 18) + j % 1024 < 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 500000000 < colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500000000 < static_cast<int64_t>(2 * pow(10, 18)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 < colLong2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((static_cast<int64_t>(2 * pow(10, 18)) + j % 2048) > (static_cast<int64_t>(2 * pow(10, 18)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 < 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 10 < 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

//FLOAT "<"
TEST(DispatcherTests, FloatLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 < 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) < 5.5)
			{
				expectedResult.push_back((float)(j % 1024 + 0.1111));
			}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5.5 < colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) > 5.5)
			{
				expectedResult.push_back((float)(j % 1024 + 0.1111));
			}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 < colFloat2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((float)(j % 1024 + 0.1111)) < ((float)(j % 2048 + 0.1111)))
			{
				expectedResult.push_back((float)(j % 1024 + 0.1111));
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 < 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((float)(j % 1024 + 0.1111));
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 10 < 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), 0);
}

//DOUBLE "<"
TEST(DispatcherTests, DoubleLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 < 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) < 5.5)
			{
				expectedResult.push_back(j % 1024 + 0.1111111);
			}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5.5 < colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) > 5.5)
			{
				expectedResult.push_back(j % 1024 + 0.1111111);
			}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 < colDouble2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024 + 0.1111111) < (j % 2048 + 0.1111111))
			{
				expectedResult.push_back(j % 1024 + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 < 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(j % 1024 + 0.1111111);
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 10 < 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), 0);
}


TEST(DispatcherTests, IntAddColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 + 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) + 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntAddColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 + 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5) > 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntAddColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 + 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5) < 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, LongAddColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 + 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) + 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongAddColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 + 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) + 5) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongAddColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 + 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) + 5) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, FloatAddColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 + 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) + 5.1111);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatAddColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 + 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5.1111) > 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatAddColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 + 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5.1111) < 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, DoubleAddColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 + 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) + 5.1111111);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleAddColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 + 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5.1111111) > 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleAddColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 + 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 5.1111111) < 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, IntSubColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 - 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) - 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntSubColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 - 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) - 5) > 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntSubColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 - 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) - 5) < 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, LongSubColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 - 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) - 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongSubColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 - 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) - 5) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongSubColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 - 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, 18)) + j % 1024) - 5) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, 18)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, FloatSubColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 - 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) + 0.1111 - 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_TRUE(std::abs(expectedResult[i] - payloads.floatpayload().floatdata()[i]) < 0.00005);
	}
}

TEST(DispatcherTests, FloatSubColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 - 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 0.1111 - 5) > 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, FloatSubColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 - 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 0.1111 - 5) < 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.floatpayload().floatdata()[i]);
	}
}

TEST(DispatcherTests, DoubleSubColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 - 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) + 0.1111111 - 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleSubColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 - 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 0.1111111 - 5) > 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleSubColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 - 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) + 0.1111111 - 5) < 500)
			{
				expectedResult.push_back((j % 1024) + 0.1111111);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}