#include <cmath>

#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/BlockBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/Database.h"
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
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByCount)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("COUNT(colInteger1)");

	ASSERT_EQ(payloads.int64payload().int64data_size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByAvg)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT AVG(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("AVG(colInteger1)");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupBySum)
{
	Context::getInstance();

	GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP BY colInteger1;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
	auto &payloads = result->payloads().at("SUM(colInteger1)");

	ASSERT_EQ(payloads.intpayload().intdata_size(), 0);
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
}
