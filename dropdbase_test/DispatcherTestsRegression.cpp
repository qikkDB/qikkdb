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
#include "../dropdbase/GpuSqlParser/ParserExceptions.h"
#include "DispatcherObjs.h"

TEST(DispatcherTestsRegression, EmptyResultFromGtColConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT colInteger1 FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	ASSERT_EQ(result->payloads().size(), 0);	// Check if the result size is also 0
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByAvg)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT AVG(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupBySum)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	ASSERT_EQ(result->payloads().size(), 0);
}


TEST(DispatcherTestsRegression, EmptySetAggregationCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptySetAggregationSum)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_EQ(result->payloads().size(), 0);
	// TODO: assert at row 0
}

TEST(DispatcherTestsRegression, EmptySetAggregationMin)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT MIN(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	ASSERT_EQ(result->payloads().size(), 0);
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

TEST(DispatcherTestsRegression, GroupByKeyOpCorrectSemantic)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT (colInteger1 + 2) * 10, COUNT(colFloat1) FROM TableA GROUP BY colInteger1 + 2;");
	ASSERT_NO_THROW(parser.parse());
}


TEST(DispatcherTestsRegression, GroupByKeyOpWrongSemantic)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT (10 * colInteger1) + 2, COUNT(colFloat1) FROM TableA GROUP BY colInteger1 + 2;");
	ASSERT_THROW(parser.parse(), ColumnGroupByException);

	GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database, "SELECT colInteger1 + 3, COUNT(colFloat1) FROM TableA GROUP BY colInteger1 + 2;");
	ASSERT_THROW(parser2.parse(), ColumnGroupByException);
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

TEST(DispatcherTestsRegression, SameAliasAsColumn)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT colInteger1 as colInteger1 FROM TableA WHERE colInteger1 > 20;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	
}
