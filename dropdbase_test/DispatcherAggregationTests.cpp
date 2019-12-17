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

    void CountIntGeneric(std::vector<int32_t> colA, std::vector<int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        aggregationDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            aggregationDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        // Execute the query_
        GpuSqlCustomParser parser(aggregationDatabase, "SELECT COUNT(colA) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at("COUNT(colA)");

        ASSERT_EQ(expectedResult.size(), payload.int64payload().int64data_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.int64payload().int64data_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payload.int64payload().int64data()[i]);
        }
    }

    void AggregationFloatGeneric(std::vector<float> colA, std::string aggregation, std::vector<float> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_FLOAT));
        aggregationDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<float>*>(
            aggregationDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        // Execute the query_
        GpuSqlCustomParser parser(aggregationDatabase,
                                  "SELECT " + aggregation + "(colA) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at(aggregation + "(colA)");

        ASSERT_EQ(expectedResult.size(), payload.floatpayload().floatdata_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.floatpayload().floatdata_size(); i++)
        {
            ASSERT_FLOAT_EQ(expectedResult[i], payload.floatpayload().floatdata()[i]);
        }
    }

    void CountFloatGeneric(std::vector<float> colA, std::vector<int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_FLOAT));
        aggregationDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<float>*>(
            aggregationDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        // Execute the query_
        GpuSqlCustomParser parser(aggregationDatabase, "SELECT COUNT(colA) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at("COUNT(colA)");

        ASSERT_EQ(expectedResult.size(), payload.int64payload().int64data_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.int64payload().int64data_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payload.int64payload().int64data()[i]);
        }
    }

    void CountSumFloatGeneric(std::vector<float> colA, std::vector<int64_t> expectedCount, std::vector<float> expectedSum)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_FLOAT));
        aggregationDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<float>*>(
            aggregationDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        // Execute the query_
        GpuSqlCustomParser parser(aggregationDatabase, "SELECT COUNT(colA), SUM(colA) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCount = result->payloads().at("COUNT(colA)");
        auto& payloadSum = result->payloads().at("SUM(colA)");

        ASSERT_EQ(expectedCount.size(), payloadCount.int64payload().int64data_size());
        ASSERT_EQ(expectedSum.size(), payloadSum.floatpayload().floatdata_size());

        for (int32_t i = 0; i < payloadCount.int64payload().int64data_size(); i++)
        {
            ASSERT_EQ(expectedCount[i], payloadCount.int64payload().int64data()[i]);
        }

        for (int32_t i = 0; i < payloadSum.floatpayload().floatdata_size(); i++)
        {
            ASSERT_EQ(expectedSum[i], payloadSum.floatpayload().floatdata()[i]);
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

TEST_F(DispatcherAggregationTests, AggCountInt)
{
    std::vector<int32_t> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    CountIntGeneric(vec, {static_cast<int64_t>(vec.size())});
}

TEST_F(DispatcherAggregationTests, AggMinFloat)
{
    std::vector<float> vec = {7.5, 6.6, 5.8, 4.9, 3.7, 2.3, 1.1, 0.6};
    std::vector<float>::iterator min = std::min_element(vec.begin(), vec.end());
    AggregationFloatGeneric(vec, "MIN", {*min});
}

TEST_F(DispatcherAggregationTests, AggMaxFloat)
{
    std::vector<float> vec = {7.5, 6.6, 5.8, 4.9, 3.7, 2.3, 1.1, 0.6};
    std::vector<float>::iterator max = std::max_element(vec.begin(), vec.end());
    AggregationFloatGeneric(vec, "MAX", {*max});
}

TEST_F(DispatcherAggregationTests, AggSumFloat)
{
    std::vector<float> vec = {7.5, 6.6, 5.8, 4.9, 3.7, 2.3, 1.1, 0.6};
    float sum = 0;
    for (auto& i : vec)
    {
        sum += i;
    }
    AggregationFloatGeneric(vec, "SUM", {sum});
}

TEST_F(DispatcherAggregationTests, AggCountFloat)
{
    std::vector<float> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    CountFloatGeneric(vec, {static_cast<int64_t>(vec.size())});
}

TEST_F(DispatcherAggregationTests, AggCountSumFloat)
{
    std::vector<float> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    float sum = 0;
    for (auto& i : vec)
    {
        sum += i;
    }
    CountSumFloatGeneric(vec, {static_cast<int64_t>(vec.size())}, {sum});
}