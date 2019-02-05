#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"

std::shared_ptr<Database> database = DatabaseGenerator::GenerateDatabase("TestDb", 1, 1 << 10);

TEST(IILExecutorTests, GtColumnConst)
{
	Context::getInstance();

	GpuSqlCustomParser parser(database, "SELECT colInteger1 FROM TableA WHERE colInteger1 > 5;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	std::vector<int32_t> expectedResult;

	for (int i = 0; i < (1 << 10); i++)
	{
		if ((i % 1024) > 5)
		{
			expectedResult.push_back(i % 1024);
		}
	}

	auto &payloads = result->payloads().at("TableA.colInteger1");

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
	}
}