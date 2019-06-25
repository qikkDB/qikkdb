#include <cmath>

#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/BlockBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "DispatcherObjs.h"

TEST(DispatcherTestsRegression, EmptyResultFromGtColConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT colInteger1 FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("TableA.colInteger1");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);		// Check if the result size is also 0

	FAIL();
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colInteger1)");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);

	FAIL();
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByAvg)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT AVG(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("AVG(colInteger1)");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);

	FAIL();
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupBySum)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("SUM(colInteger1)");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);

	FAIL();
}


TEST(DispatcherTestsRegression, EmptySetAggregationCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colInteger1)");
	
	ASSERT_EQ(payloads.int64payload().int64data_size(), 1);	// Check if the result size is 1
	ASSERT_EQ(payloads.int64payload().int64data()[0], 0);	// and at row 0 is count = 0

	FAIL();
}

TEST(DispatcherTestsRegression, EmptySetAggregationSum)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("SUM(colInteger1)");
	ASSERT_EQ(payloads.intpayload().intdata_size(), 1);	// Check if the result size is 1
	// TODO: assert at row 0

	FAIL();
}

TEST(DispatcherTestsRegression, EmptySetAggregationMin)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT MIN(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("MIN(colInteger1)");
	ASSERT_EQ(payloads.intpayload().intdata_size(), 1);	// Check if the result size is 1
	// TODO: assert at row 0

	FAIL();
}

TEST(DispatcherTestsRegression, PointAggregationCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colPoint1) FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colPoint1)");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
	ASSERT_EQ(payloads.int64payload().int64data()[0], TEST_BLOCK_COUNT * TEST_BLOCK_SIZE);
}

TEST(DispatcherTestsRegression, PointAggregationCountWithWhere)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colPoint1) FROM TableA WHERE colInteger1 > 0;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colPoint1)");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
	// Count sufficient row on CPU
	auto columnInt = dynamic_cast<ColumnBase<int32_t>*>(DispatcherObjs::GetInstance().database->GetTables().at("TableA").GetColumns().at("colInteger1").get());
	int32_t expectedCount = 0;
	for (int i = 0; i < TEST_BLOCK_COUNT; i++)
	{
		auto blockInt = columnInt->GetBlocksList()[i];
		for (int k = 0; k < TEST_BLOCK_SIZE; k++)
		{
			if (blockInt->GetData()[k] > 0)
			{
				expectedCount++;
			}
		}
	}

	ASSERT_EQ(payloads.int64payload().int64data()[0], expectedCount);
}

TEST(DispatcherTestsRegression, Int32AggregationCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colInteger1)");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
	ASSERT_EQ(payloads.int64payload().int64data()[0], TEST_BLOCK_COUNT * TEST_BLOCK_SIZE);
}

TEST(DispatcherTestsRegression, ConstOpOnMultiGPU)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT 2+2 FROM TableA;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("2+2");
	ASSERT_EQ(payloads.intpayload().intdata_size(), 1);
	ASSERT_EQ(payloads.intpayload().intdata()[0], 4);

}
