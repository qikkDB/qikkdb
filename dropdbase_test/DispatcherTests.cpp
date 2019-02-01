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
			if (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024 > 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if (500000000 > static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if ((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048) > (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048);
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
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if (500000000 < static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if ((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048) > (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

/////////////////////
//   ">=" operator
/////////////////////

//INT ">="
TEST(DispatcherTests, IntEqGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 >= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024) >= 5)
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

TEST(DispatcherTests, IntEqGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 500 >= colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500 >= j % 1024)
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

TEST(DispatcherTests, IntEqGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger2 FROM TableA WHERE colInteger2 >= colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048) >= (j % 1024))
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

TEST(DispatcherTests, IntEqGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 10 >= 5;");
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

TEST(DispatcherTests, IntEqGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 >= 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

// LONG ">="
TEST(DispatcherTests, LongEqGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 >= 500000000;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024 >= 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 500000000 >= colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500000000 >= static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongEqGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong2 FROM TableA WHERE colLong2 >= colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048) >= (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048);
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

TEST(DispatcherTests, LongEqGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 10 >= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 >= 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

//FLOAT ">="
TEST(DispatcherTests, FloatEqGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 >= 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) >= 5.5)
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

TEST(DispatcherTests, FloatEqGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5.5 >= colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) <= 5.5)
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

TEST(DispatcherTests, FloatEqGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat2 FROM TableA WHERE colFloat2 >= colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((float)(j % 2048 + 0.1111)) >= ((float)(j % 1024 + 0.1111)))
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

TEST(DispatcherTests, FloatEqGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 10 >= 5;");
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

TEST(DispatcherTests, FloatEqGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 >= 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), 0);
}

//DOUBLE ">="
TEST(DispatcherTests, DoubleEqGtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 >= 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) >= 5.5)
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

TEST(DispatcherTests, DoubleEqGtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5.5 >= colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) <= 5.5)
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

TEST(DispatcherTests, DoubleEqGtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble2 FROM TableA WHERE colDouble2 >= colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048 + 0.1111111) >= (j % 1024 + 0.1111111))
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

TEST(DispatcherTests, DoubleEqGtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 10 >= 5;");
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

TEST(DispatcherTests, DoubleEqGtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 >= 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), 0);
}

/////////////////////
//   "<=" operator
/////////////////////

//INT "<="
TEST(DispatcherTests, IntEqLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 <= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024) <= 5)
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

TEST(DispatcherTests, IntEqLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 500 <= colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500 <= j % 1024)
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

TEST(DispatcherTests, IntEqLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 <= colInteger2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024) <= (j % 2048))
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

TEST(DispatcherTests, IntEqLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 <= 10;");
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

TEST(DispatcherTests, IntEqLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 10 <= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

// LONG "<="
TEST(DispatcherTests, LongEqLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 <= 500000000;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (2 * (10, 18) + j % 1024 <= 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 500000000 <= colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500000000 <= static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongEqLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 <= colLong2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048) >= (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongEqLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 <= 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 10 <= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

//FLOAT "<="
TEST(DispatcherTests, FloatEqLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 <= 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) <= 5.5)
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

TEST(DispatcherTests, FloatEqLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5.5 <= colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) >= 5.5)
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

TEST(DispatcherTests, FloatEqLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 <= colFloat2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((float)(j % 1024 + 0.1111)) <= ((float)(j % 2048 + 0.1111)))
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

TEST(DispatcherTests, FloatEqLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 <= 10;");
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

TEST(DispatcherTests, FloatEqLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 10 <= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), 0);
}

//DOUBLE "<="
TEST(DispatcherTests, DoubleEqLtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 <= 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) <= 5.5)
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

TEST(DispatcherTests, DoubleEqLtConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5.5 <= colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) >= 5.5)
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

TEST(DispatcherTests, DoubleEqLtColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 <= colDouble2;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024 + 0.1111111) <= (j % 2048 + 0.1111111))
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

TEST(DispatcherTests, DoubleEqLtConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 <= 10;");
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

TEST(DispatcherTests, DoubleEqLtConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 10 <= 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), 0);
}


/////////////////////
//   "=" operator
/////////////////////

//INT "="
TEST(DispatcherTests, IntEqColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 = 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024) == 5)
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

TEST(DispatcherTests, IntEqConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 500 = colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500 == j % 1024)
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

TEST(DispatcherTests, IntEqColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger2 FROM TableA WHERE colInteger2 = colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048) == (j % 1024))
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

TEST(DispatcherTests, IntEqConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 = 5;");
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

TEST(DispatcherTests, IntEqConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE 5 = 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

// LONG ">"
TEST(DispatcherTests, LongEqColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 = 500000000;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024 == 500000000)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
			}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 500000000 = colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (500000000 == static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongEqColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong2 FROM TableA WHERE colLong2 = colLong1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048) == (static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024))
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 2048);
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

TEST(DispatcherTests, LongEqConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 = 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongEqConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE 5 = 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;

	auto &payloads = result->payloads().at("TableA.colLong1");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

//FLOAT ">"
TEST(DispatcherTests, FloatEqColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 = 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) == 5.5)
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

TEST(DispatcherTests, FloatEqConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5.5 = colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if (((float)(j % 1024 + 0.1111)) == 5.5)
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

TEST(DispatcherTests, FloatEqColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat2 FROM TableA WHERE colFloat2 = colFloat1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((float)(j % 2048 + 0.1111)) == ((float)(j % 1024 + 0.1111)))
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

TEST(DispatcherTests, FloatEqConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 = 5;");
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

TEST(DispatcherTests, FloatEqConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE 5 = 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;

	auto &payloads = result->payloads().at("TableA.colFloat1");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), 0);
}

//DOUBLE ">"
TEST(DispatcherTests, DoubleEqColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 = 5.5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) == 5.5)
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

TEST(DispatcherTests, DoubleEqConstColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5.5 = colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
			if ((j % 1024 + 0.1111111) == 5.5)
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

TEST(DispatcherTests, DoubleEqColumnColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble2 FROM TableA WHERE colDouble2 = colDouble1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 2048 + 0.1111111) == (j % 1024 + 0.1111111))
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

TEST(DispatcherTests, DoubleEqConstConstTrue)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 = 5;");
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

TEST(DispatcherTests, DoubleEqConstConstFalse)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE 5 = 10;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;

	auto &payloads = result->payloads().at("TableA.colDouble1");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), 0);
}

///////////

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
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) + 5);
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
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) + 5) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) + 5) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) - 5);
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
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) - 5) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) - 5) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

//multiply tests:
TEST(DispatcherTests, IntMulColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 * 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) * 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntMulColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 * 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) * 5) > 500)
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

TEST(DispatcherTests, IntMulColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 * 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((j % 1024) * 5) < 500)
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

TEST(DispatcherTests, LongMulColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 * 2 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) * 2);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongMulColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 * 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) * 5) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongMulColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 * 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) * 5) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, FloatMulColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 * 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(((j % 1024) + 0.1111) * 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_TRUE(std::abs(expectedResult[i] - payloads.floatpayload().floatdata()[i]) < 0.0005);
	}
}

TEST(DispatcherTests, FloatMulColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 * 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111) * 5) > 500)
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

TEST(DispatcherTests, FloatMulColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 * 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111) * 5) < 500)
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

TEST(DispatcherTests, DoubleMulColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 * 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(((j % 1024) + 0.1111111) * 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleMulColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 * 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111111) * 5) > 500)
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

TEST(DispatcherTests, DoubleMulColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 * 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111111) * 5) < 500)
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

//divide tests:
TEST(DispatcherTests, IntDivColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 / 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int32_t>((j % 1024) / 5));
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntDivColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 / 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (static_cast<int32_t>((j % 1024) / 5) > 500)
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

TEST(DispatcherTests, IntDivColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 / 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (static_cast<int32_t>((j % 1024) / 5) < 500)
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

TEST(DispatcherTests, LongDivColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 / 2 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(static_cast<int64_t>((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 2));
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongDivColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 / 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (static_cast<int64_t>(((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 5)) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, LongDivColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 / 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (static_cast<int64_t>(((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 5)) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
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

TEST(DispatcherTests, FloatDivColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 / 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(((j % 1024) + 0.1111) / 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_TRUE(std::abs(expectedResult[i] - payloads.floatpayload().floatdata()[i]) < 0.0005);
	}
}

TEST(DispatcherTests, FloatDivColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 / 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111) / 5) > 500)
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

TEST(DispatcherTests, FloatDivColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colFloat1 FROM TableA WHERE colFloat1 / 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111) / 5) < 500)
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

TEST(DispatcherTests, DoubleDivColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 / 5 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back(((j % 1024) + 0.1111111) / 5);
		}
	}

	auto &payloads = result->payloads().at("R0");

	ASSERT_EQ(payloads.doublepayload().doubledata_size(), expectedResult.size());

	for (int i = 0; i < payloads.doublepayload().doubledata_size(); i++)
	{
		ASSERT_DOUBLE_EQ(expectedResult[i], payloads.doublepayload().doubledata()[i]);
	}
}

TEST(DispatcherTests, DoubleDivColumnConstGtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 / 5 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111111) / 5) > 500)
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

TEST(DispatcherTests, DoubleDivColumnConstLtConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colDouble1 FROM TableA WHERE colDouble1 / 5 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<double> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((((j % 1024) + 0.1111111) / 5) < 500)
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

////////////////////////////////////////////////
///////////////////////////////////////////////

TEST(DispatcherTests, IntDivColumnConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 / 5.0 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((j % 1024) / 5.0);
		}
	}

	auto &payloads = result->payloads().at("R0");

//	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntDivColumnConstGtConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 / 5.0 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024) / 5.0 > 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

//	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, IntDivColumnConstLtConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 / 5.0 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if ((j % 1024) / 5.0 < 500)
			{
				expectedResult.push_back(j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

//	ASSERT_EQ(payloads.intpayload().intdata_size(), expectedResult.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}

TEST(DispatcherTests, LongDivColumnConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 / 2.0 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<float> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			expectedResult.push_back((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 2.0);
		}
	}

	auto &payloads = result->payloads().at("R0");

//	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_FLOAT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongDivColumnConstGtConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 / 5.0 > 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 5.0) > 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

//	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}

TEST(DispatcherTests, LongDivColumnConstLtConstFloat)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colLong1 FROM TableA WHERE colLong1 / 5.0 < 500;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int64_t> expectedResult;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (1 << 11); j++)
		{
			if (((static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024) / 5.0) < 500)
			{
				expectedResult.push_back(static_cast<int64_t>(2 * pow(10, j % 19)) + j % 1024);
			}
		}
	}

	auto &payloads = result->payloads().at("TableA.colLong1");

//	ASSERT_EQ(payloads.int64payload().int64data_size(), expectedResult.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.int64payload().int64data()[i]);
	}
}