#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"


class DispatcherOrderByTests : public ::testing::Test
{
protected:
    const std::string path = Configuration::GetInstance().GetDatabaseDir();
    const std::string dbName = "OrderByTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> orderByDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        orderByDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(orderByDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    void OrderByWhereIntIntGeneric(std::vector<int32_t> colA,
                                   std::vector<int32_t> colB,
                                   int32_t threshold,
                                   std::vector<int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        orderByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            orderByDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        reinterpret_cast<ColumnBase<int32_t>*>(
            orderByDatabase->GetTables().at(tableName).GetColumns().at("colB").get())
            ->InsertData(colB);

        // Execute the query_
        GpuSqlCustomParser parser(orderByDatabase, "SELECT colA FROM " + tableName + " WHERE colB > " +
                                                       std::to_string(threshold) + " ORDER BY colA;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at(tableName + ".colA");

        ASSERT_EQ(expectedResult.size(), payload.intpayload().intdata_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.intpayload().intdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payload.intpayload().intdata()[i]);
        }
    }

    void OrderByWhereIntIntStringGeneric(std::vector<int32_t> colA,
                                         std::vector<int32_t> colB,
                                         std::vector<std::string> colC,
                                         int32_t threshold,
                                         std::vector<std::string> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_STRING));
        orderByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            orderByDatabase->GetTables().at(tableName).GetColumns().at("colA").get())
            ->InsertData(colA);

        reinterpret_cast<ColumnBase<int32_t>*>(
            orderByDatabase->GetTables().at(tableName).GetColumns().at("colB").get())
            ->InsertData(colB);

        reinterpret_cast<ColumnBase<std::string>*>(
            orderByDatabase->GetTables().at(tableName).GetColumns().at("colC").get())
            ->InsertData(colC);

        // Execute the query_
        GpuSqlCustomParser parser(orderByDatabase, "SELECT colC FROM " + tableName + " WHERE colB > " +
                                                       std::to_string(threshold) + " ORDER BY colA;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payload = result->payloads().at(tableName + ".colC");

        ASSERT_EQ(expectedResult.size(), payload.stringpayload().stringdata_size())
            << " wrong number of results";
        for (int32_t i = 0; i < payload.stringpayload().stringdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payload.stringpayload().stringdata()[i]);
        }
    }
};

TEST_F(DispatcherOrderByTests, OrderByWhereIntInt)
{
    OrderByWhereIntIntGeneric({7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}, 3, {0, 1, 2, 3});
}

TEST_F(DispatcherOrderByTests, OrderByWhereIntIntString)
{
    OrderByWhereIntIntStringGeneric({7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
                                    {"a", "b", "c", "d", "e", "f", "g", "h"}, 3, {"h", "g", "f", "e"});
}