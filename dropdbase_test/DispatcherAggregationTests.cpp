#include <algorithm>
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

class DispatcherAggregationTests : public ::testing::Test
{
protected:
    const std::string dbName = "AggregationTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> aggregationDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        aggregationDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(aggregationDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    void AggregationIntGeneric(std::vector<int32_t> colA, std::string aggregation, std::vector<int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        aggregationDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            aggregationDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        // Execute the query_
        GpuSqlCustomParser parser(aggregationDatabase,
                                  "SELECT " + aggregation + "(colA) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at(aggregation + "(colA)");

        ASSERT_EQ(expectedResult.size(), payload.intpayload().intdata_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.intpayload().intdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payload.intpayload().intdata()[i]);
        }
    }
};

TEST_F(DispatcherAggregationTests, AggMinInt)
{
    std::vector<int32_t> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<int32_t>::iterator min = std::min_element(vec.begin(), vec.end());
    AggregationIntGeneric(vec, "MIN", {*min});
}

TEST_F(DispatcherAggregationTests, AggMaxInt)
{
    std::vector<int32_t> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<int32_t>::iterator max = std::max_element(vec.begin(), vec.end());
    AggregationIntGeneric(vec, "MAX", {*max});
}

TEST_F(DispatcherAggregationTests, AggSumInt)
{
    std::vector<int32_t> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    int32_t sum = 0;
    for (auto& i : vec)
    {
        sum += i;
    }
    AggregationIntGeneric(vec, "SUM", {sum});
}