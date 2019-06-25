#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

// This test is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS colPoint;"

class DispatcherGroupByTests : public ::testing::Test
{
protected:
    const std::string dbName = "GroupByTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> groupByDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        groupByDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(groupByDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    // This is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS
    // colPoint;" polygon - const, wkt from query; point - col (as vector of NativeGeoPoints here)
    void GBSGenericTest(std::vector<std::string> keys,
                        std::vector<int32_t> values,
                        std::unordered_map<std::string, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        // Create column with IDs
        std::vector<int32_t> colID;
        for (int i = 0; i < keys.size(); i++)
        {
            colID.push_back(i);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colID").get())
            ->InsertData(colID);
        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, SUM(colInteger) FROM " +
                                                       tableName + " GROUP BY colString;");
        auto resultPtr = parser.parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at("SUM(colInteger)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }
};

TEST_F(DispatcherGroupByTests, StringSimpleSum)
{
    GBSGenericTest({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                   {{"Apple", 4}, {"Abcd", 9}, {"Banana", 5}, {"XYZ", 38}, {"0", 10}});
}
