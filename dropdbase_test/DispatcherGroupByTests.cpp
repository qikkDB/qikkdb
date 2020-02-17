#include <boost/functional/hash.hpp>

#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

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

    void GroupByIntGenericTest(std::string aggregationFunction,
                               std::vector<int32_t> keys,
                               std::vector<int32_t> values,
                               std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, " + aggregationFunction + "(colIntegerV) FROM " +
                                                       tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByIntAliasGenericTest(std::string aggregationFunction,
                                    std::vector<int32_t> keys,
                                    std::vector<int32_t> values,
                                    std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, " + aggregationFunction +
                                                       "(colIntegerV) FROM " + tableName + " GROUP BY 1;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByValueOpIntGenericTest(std::string aggregationFunction,
                                      std::vector<int32_t> keys,
                                      std::vector<int32_t> values,
                                      std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, " + aggregationFunction + "(colIntegerV - 2) FROM " +
                                                       tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("SimpleTable.colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV-2)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByKeyOpIntGenericTest(std::string aggregationFunction,
                                    std::vector<int32_t> keys,
                                    std::vector<int32_t> values,
                                    std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK + 2, " + aggregationFunction +
                                                       "(colIntegerV) FROM " + tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("colIntegerK+2");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByKeyWhereIntGenericTest(std::string aggregationFunction,
                                       std::vector<int32_t> keys,
                                       std::vector<int32_t> values,
                                       int32_t threshold,
                                       std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase,
                                  "SELECT colIntegerK, " + aggregationFunction + "(colIntegerV) FROM " +
                                      tableName + " WHERE colIntegerK = " + std::to_string(threshold) +
                                      " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("SimpleTable.colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByKeyOpWhereIntGenericTest(std::string aggregationFunction,
                                         std::vector<int32_t> keys,
                                         std::vector<int32_t> values,
                                         int32_t threshold,
                                         std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase,
                                  "SELECT colIntegerK, " + aggregationFunction + "(colIntegerV) FROM " +
                                      tableName + " WHERE colIntegerK + 2 = " + std::to_string(threshold) +
                                      " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("SimpleTable.colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByKeyWhereDelimitedAliasIntGenericTest(std::string aggregationFunction,
                                                     std::vector<int32_t> keys,
                                                     std::vector<int32_t> values,
                                                     int32_t threshold,
                                                     std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase,
                                  "SELECT [colIntegerK] as [Dimension2], " + aggregationFunction +
                                      "(colIntegerV) FROM " + tableName + " WHERE [Dimension2] = " +
                                      std::to_string(threshold) + " GROUP BY [Dimension2];");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("Dimension2");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByKeyAggOpIntGenericTest(std::string aggregationFunction,
                                       std::vector<int32_t> keys,
                                       std::unordered_map<int32_t, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, " + aggregationFunction + "(colIntegerK - 2) FROM " +
                                                       tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("SimpleTable.colIntegerK");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerK-2)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }
    void GroupByIntCountTest(std::vector<int32_t> keys,
                             std::vector<int32_t> values,
                             std::unordered_map<int32_t, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, COUNT(colIntegerV) FROM " +
                                                       tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colIntegerK");
        auto& payloadValues = result->payloads().at("COUNT(colIntegerV)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GroupByIntCountAsteriskTest(std::vector<int32_t> keys,
                                     std::vector<int32_t> values,
                                     std::unordered_map<int32_t, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colIntegerK", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerK").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colIntegerK, COUNT(*) FROM " +
                                                       tableName + " GROUP BY colIntegerK;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colIntegerK");
        auto& payloadValues = result->payloads().at("COUNT(*)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }

    //== These tests are for testing queries with GROUP BY String column
    void GBSGenericTest(std::string aggregationFunction,
                        std::vector<std::string> keys,
                        std::vector<int32_t> values,
                        std::unordered_map<std::string, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, " + aggregationFunction + "(colInteger) FROM " +
                                                       tableName + " GROUP BY colString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colInteger)");

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

    void GBSOBGenericTest(std::string aggregationFunction,
                          std::vector<std::string> keys,
                          std::vector<int32_t> values,
                          std::vector<std::pair<std::string, int32_t>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, " + aggregationFunction + "(colInteger) FROM " +
                                                       tableName + " GROUP BY colString ORDER BY " +
                                                       aggregationFunction + "(colInteger);");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colInteger)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_EQ(expectedResult[i].first, key) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[i].second, payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GBSCountTest(std::vector<std::string> keys,
                      std::vector<int32_t> values,
                      std::unordered_map<std::string, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, COUNT(colInteger) FROM " +
                                                       tableName + " GROUP BY colString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at("COUNT(colInteger)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GBSCountStringWhereTest(std::vector<std::string> keys,
                                 std::vector<int32_t> values,
                                 std::unordered_map<std::string, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, COUNT(colString) FROM " + tableName +
                                                       " WHERE colInteger > 5 GROUP BY colString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at("COUNT(colString)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GBSCountStringTest(std::vector<std::string> keys, std::unordered_map<std::string, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colString, COUNT(colString) FROM " +
                                                       tableName + " GROUP BY colString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at("COUNT(colString)");

        // ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
        //   << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            std::cout << (expectedResult.find(key) == expectedResult.end()) << " key \"" << key
                      << "\"" << std::endl;
            ;
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GBSKeyOpGenericTest(std::string aggregationFunction,
                             std::vector<std::string> keys,
                             std::vector<int32_t> values,
                             std::unordered_map<std::string, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT LEFT(colString, 3), " + aggregationFunction +
                                                       "(colInteger) FROM " + tableName + " GROUP BY colString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at("LEFT(colString,3)");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colInteger)");

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

    void GBOBKeysGenericTest(std::string aggregationFunction,
                             std::vector<int32_t> keys,
                             std::vector<std::pair<int32_t, int32_t>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
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
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(keys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colInteger, " + aggregationFunction +
                                                       "(colID) FROM " + tableName +
                                                       " GROUP BY colInteger ORDER BY colInteger;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colInteger");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colID)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_EQ(expectedResult[i].first, key) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[i].second, payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    void GBOBValuesGenericTest(std::string aggregationFunction,
                               std::vector<std::string> keys,
                               std::vector<int32_t> values,
                               std::vector<std::pair<int32_t, int32_t>> expectedResult)
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

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colInteger, " + aggregationFunction + "(colID) FROM " +
                                                       tableName + " GROUP BY colInteger ORDER BY " +
                                                       aggregationFunction + "(colID) - 2;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colInteger");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colID)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_EQ(expectedResult[i].first, key) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[i].second, payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

    //== These tests are for GroupBy multi-key
    void GroupByMultiKeyGenericTest(std::string aggregationFunction,
                                    std::vector<std::vector<int32_t>> keys,
                                    std::vector<int32_t> values,
                                    std::unordered_map<std::vector<int32_t>, int32_t, boost::hash<std::vector<int32_t>>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        for (int32_t i = 0; i < keys.size(); i++)
        {
            columns.insert(std::make_pair<std::string, DataType>("colIntegerK" + std::to_string(i),
                                                                 DataType::COLUMN_INT));
        }
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        for (int32_t i = 0; i < keys.size(); i++)
        {
            reinterpret_cast<ColumnBase<int32_t>*>(groupByDatabase->GetTables()
                                                       .at(tableName)
                                                       .GetColumns()
                                                       .at("colIntegerK" + std::to_string(i))
                                                       .get())
                ->InsertData(keys[i]);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        std::string multiCols;
        for (int32_t i = 0; i < keys.size(); i++)
        {
            multiCols += "colIntegerK" + std::to_string(i) + (i == keys.size() - 1 ? "" : ", ");
        }
        std::cout << "Running GroupBy multi-key: " << multiCols << std::endl;
        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT " + multiCols + ", " +
                                                       aggregationFunction + "(colIntegerV) FROM " +
                                                       tableName + " GROUP BY " + multiCols + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

        std::vector<ColmnarDB::NetworkClient::Message::QueryResponsePayload> payloadKeys;

        for (int32_t i = 0; i < keys.size(); i++)
        {
            payloadKeys.emplace_back(result->payloads().at(tableName + ".colIntegerK" + std::to_string(i)));
        }
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        for (int32_t i = 0; i < keys.size(); i++)
        {
            ASSERT_EQ(expectedResult.size(), payloadKeys[i].intpayload().intdata_size())
                << " wrong number of keys at col " << i;
        }

        for (int32_t i = 0; i < payloadKeys[0].intpayload().intdata_size(); i++)
        {
            std::vector<int32_t> key;
            for (int32_t c = 0; c < payloadKeys.size(); c++)
            {
                key.emplace_back(payloadKeys[c].intpayload().intdata()[i]);
                std::cout << payloadKeys[c].intpayload().intdata()[i]
                          << (c == payloadKeys.size() - 1 ? ": " : ", ");
            }
            std::cout << payloadValues.intpayload().intdata()[i] << std::endl;

            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at result row " << i;
        }
    }

    void GroupByMultiKeyCountTest(std::vector<std::vector<int32_t>> keys,
                                  std::vector<int32_t> values,
                                  std::unordered_map<std::vector<int32_t>, int64_t, boost::hash<std::vector<int32_t>>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        for (int32_t i = 0; i < keys.size(); i++)
        {
            columns.insert(std::make_pair<std::string, DataType>("colIntegerK" + std::to_string(i),
                                                                 DataType::COLUMN_INT));
        }
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        for (int32_t i = 0; i < keys.size(); i++)
        {
            reinterpret_cast<ColumnBase<int32_t>*>(groupByDatabase->GetTables()
                                                       .at(tableName)
                                                       .GetColumns()
                                                       .at("colIntegerK" + std::to_string(i))
                                                       .get())
                ->InsertData(keys[i]);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        std::string multiCols;
        for (int32_t i = 0; i < keys.size(); i++)
        {
            multiCols += "colIntegerK" + std::to_string(i) + (i == keys.size() - 1 ? "" : ", ");
        }
        std::cout << "Running GroupBy multi-key: " << multiCols << std::endl;
        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT " + multiCols + ", COUNT(colIntegerV) FROM " +
                                                       tableName + " GROUP BY " + multiCols + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

        std::vector<ColmnarDB::NetworkClient::Message::QueryResponsePayload> payloadKeys;

        for (int32_t i = 0; i < keys.size(); i++)
        {
            payloadKeys.emplace_back(result->payloads().at(tableName + ".colIntegerK" + std::to_string(i)));
        }
        auto& payloadValues = result->payloads().at("COUNT(colIntegerV)");

        for (int32_t i = 0; i < keys.size(); i++)
        {
            ASSERT_EQ(expectedResult.size(), payloadKeys[i].intpayload().intdata_size())
                << " wrong number of keys at col " << i;
        }

        for (int32_t i = 0; i < payloadKeys[0].intpayload().intdata_size(); i++)
        {
            std::vector<int32_t> key;
            for (int32_t c = 0; c < payloadKeys.size(); c++)
            {
                key.emplace_back(payloadKeys[c].intpayload().intdata()[i]);
                std::cout << payloadKeys[c].intpayload().intdata()[i]
                          << (c == payloadKeys.size() - 1 ? ": " : ", ");
            }
            std::cout << payloadValues.int64payload().int64data()[i] << std::endl;

            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at result row " << i;
        }
    }

    void GroupByMultiKeyCountAsteriskTest(
        std::vector<std::vector<int32_t>> keys,
        std::vector<int32_t> values,
        std::unordered_map<std::vector<int32_t>, int64_t, boost::hash<std::vector<int32_t>>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        for (int32_t i = 0; i < keys.size(); i++)
        {
            columns.insert(std::make_pair<std::string, DataType>("colIntegerK" + std::to_string(i),
                                                                 DataType::COLUMN_INT));
        }
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        for (int32_t i = 0; i < keys.size(); i++)
        {
            reinterpret_cast<ColumnBase<int32_t>*>(groupByDatabase->GetTables()
                                                       .at(tableName)
                                                       .GetColumns()
                                                       .at("colIntegerK" + std::to_string(i))
                                                       .get())
                ->InsertData(keys[i]);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        std::string multiCols;
        for (int32_t i = 0; i < keys.size(); i++)
        {
            multiCols += "colIntegerK" + std::to_string(i) + (i == keys.size() - 1 ? "" : ", ");
        }
        std::cout << "Running GroupBy multi-key: " << multiCols << std::endl;
        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT " + multiCols + ", COUNT(*) FROM " +
                                                       tableName + " GROUP BY " + multiCols + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

        std::vector<ColmnarDB::NetworkClient::Message::QueryResponsePayload> payloadKeys;

        for (int32_t i = 0; i < keys.size(); i++)
        {
            payloadKeys.emplace_back(result->payloads().at(tableName + ".colIntegerK" + std::to_string(i)));
        }
        auto& payloadValues = result->payloads().at("COUNT(*)");

        for (int32_t i = 0; i < keys.size(); i++)
        {
            ASSERT_EQ(expectedResult.size(), payloadKeys[i].intpayload().intdata_size())
                << " wrong number of keys at col " << i;
        }

        for (int32_t i = 0; i < payloadKeys[0].intpayload().intdata_size(); i++)
        {
            std::vector<int32_t> key;
            for (int32_t c = 0; c < payloadKeys.size(); c++)
            {
                key.emplace_back(payloadKeys[c].intpayload().intdata()[i]);
                std::cout << payloadKeys[c].intpayload().intdata()[i]
                          << (c == payloadKeys.size() - 1 ? ": " : ", ");
            }
            std::cout << payloadValues.int64payload().int64data()[i] << std::endl;

            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at result row " << i;
        }
    }

    void GroupByMultiKeyStringTest(
        std::string aggregationFunction,
        std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<std::string>> keys,
        std::vector<int32_t> values,
        std::unordered_map<std::tuple<int32_t, int32_t, std::string>, int32_t, boost::hash<std::tuple<int32_t, int32_t, std::string>>> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colKeyInt0", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKeyInt1", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKeyString0", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colIntegerV", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeyInt0").get())
            ->InsertData(std::get<0>(keys));
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeyInt1").get())
            ->InsertData(std::get<1>(keys));
        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeyString0").get())
            ->InsertData(std::get<2>(keys));
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colIntegerV").get())
            ->InsertData(values);

        std::string multiCols = "colKeyInt0, colKeyInt1, colKeyString0";
        std::cout << "Running GroupBy multi-key: " << multiCols << std::endl;
        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT " + multiCols + ", " +
                                                       aggregationFunction + "(colIntegerV) FROM " +
                                                       tableName + " GROUP BY " + multiCols + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

        std::vector<ColmnarDB::NetworkClient::Message::QueryResponsePayload> payloadKeys;
        payloadKeys.emplace_back(result->payloads().at(tableName + ".colKeyInt0"));
        payloadKeys.emplace_back(result->payloads().at(tableName + ".colKeyInt1"));
        payloadKeys.emplace_back(result->payloads().at(tableName + ".colKeyString0"));
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colIntegerV)");

        for (int32_t i = 0; i < payloadKeys.size(); i++)
        {
            ASSERT_EQ(expectedResult.size(), i == 2 ? payloadKeys[i].stringpayload().stringdata_size() :
                                                      payloadKeys[i].intpayload().intdata_size())
                << " wrong number of keys at col " << i;
        }

        for (int32_t i = 0; i < payloadKeys[0].intpayload().intdata_size(); i++)
        {
            std::tuple<int32_t, int32_t, std::string> key{payloadKeys[0].intpayload().intdata()[i],
                                                          payloadKeys[1].intpayload().intdata()[i],
                                                          payloadKeys[2].stringpayload().stringdata()[i]};
            std::cout << std::get<0>(key) << ", ";
            std::cout << std::get<1>(key) << ", ";
            std::cout << std::get<2>(key) << ": ";
            std::cout << payloadValues.intpayload().intdata()[i] << std::endl;

            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at result row " << i;
        }
    }

    // Test for reconstruction of null masks (for both key and value) at main GroupBy
    void GroupByIntWithWhere(
        std::string aggregationFunction,
        std::vector<int32_t> filterNumbers,
        int32_t threshold,
        std::vector<int32_t> keys,
        std::vector<bool> keyNulls,
        std::vector<int32_t> values,
        std::vector<bool> valueNulls,
        std::unordered_map<std::pair<int32_t, bool>, std::pair<int32_t, bool>, boost::hash<std::pair<int32_t, bool>>> expectedResult)
    {
        ASSERT_EQ(filterNumbers.size(), keys.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), keyNulls.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), values.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), valueNulls.size()) << "bad test input sizes";

        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colFilter", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKey", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colValue", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        std::vector<int64_t> compressedKeyNullMask((keyNulls.size() + 7) / 8);
        std::vector<int64_t> compressedValueNullMask((valueNulls.size() + 7) / 8);
        for (int32_t i = 0; i < keyNulls.size(); i++)
        {
            if (i % 8 == 0)
            {
                compressedKeyNullMask[i / 8] = 0;
            }
            compressedKeyNullMask[i / 8] |= ((keyNulls[i] ? 1 : 0) << (i % 8));
        }
        for (int32_t i = 0; i < valueNulls.size(); i++)
        {
            if (i % 8 == 0)
            {
                compressedValueNullMask[i / 8] = 0;
            }
            compressedValueNullMask[i / 8] |= ((valueNulls[i] ? 1 : 0) << (i % 8));
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colFilter").get())
            ->InsertData(filterNumbers);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKey").get())
            ->InsertData(keys, compressedKeyNullMask);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colValue").get())
            ->InsertData(values, compressedValueNullMask);

        // Execute the query
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colKey, " + aggregationFunction +
                                                       "(colValue) FROM " + tableName + " WHERE colFilter > " +
                                                       std::to_string(threshold) + " GROUP BY colKey;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colKey");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colValue)");
        auto& keysNullMaskResult = result->nullbitmasks().at(tableName + ".colKey").nullmask();
        auto& valuesNullMaskResult = result->nullbitmasks().at(aggregationFunction + "(colValue)").nullmask();

        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            const char keyChar = keysNullMaskResult[i / 8];
            const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
            const char valueChar = valuesNullMaskResult[i / 8];
            const bool valueIsNull = ((valueChar >> (i % 8)) & 1);
            int32_t key = keyIsNull ? -1 : payloadKeys.intpayload().intdata()[i];
            int32_t value = valueIsNull ? -1 : payloadValues.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find({key, keyIsNull}) == expectedResult.end())
                << " key " << (keyIsNull ? "NULL" : std::to_string(key)) << " not found";

            ASSERT_EQ(std::get<0>(expectedResult.at({key, keyIsNull})), value)
                << " value at key " << (keyIsNull ? "NULL" : std::to_string(key));
            ASSERT_EQ(std::get<1>(expectedResult.at({key, keyIsNull})), valueIsNull)
                << " value null bit at key " << (keyIsNull ? "NULL" : std::to_string(key));
        }
    }

    // Test for reconstruction of null masks (for both key and value) at string GroupBy
    void GroupByStringWithWhere(
        std::string aggregationFunction,
        std::vector<int32_t> filterNumbers,
        int32_t threshold,
        std::vector<std::string> keys,
        std::vector<bool> keyNulls,
        std::vector<int32_t> values,
        std::vector<bool> valueNulls,
        std::unordered_map<std::pair<std::string, bool>, std::pair<int32_t, bool>, boost::hash<std::pair<std::string, bool>>> expectedResult)
    {
        ASSERT_EQ(filterNumbers.size(), keys.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), keyNulls.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), values.size()) << "bad test input sizes";
        ASSERT_EQ(filterNumbers.size(), valueNulls.size()) << "bad test input sizes";

        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colFilter", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKey", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colValue", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        std::vector<int64_t> compressedKeyNullMask((keyNulls.size() + 7) / 8);
        std::vector<int64_t> compressedValueNullMask((valueNulls.size() + 7) / 8);
        for (int32_t i = 0; i < keyNulls.size(); i++)
        {
            if (i % 8 == 0)
            {
                compressedKeyNullMask[i / 8] = 0;
            }
            compressedKeyNullMask[i / 8] |= ((keyNulls[i] ? 1 : 0) << (i % 8));
        }
        for (int32_t i = 0; i < valueNulls.size(); i++)
        {
            if (i % 8 == 0)
            {
                compressedValueNullMask[i / 8] = 0;
            }
            compressedValueNullMask[i / 8] |= ((valueNulls[i] ? 1 : 0) << (i % 8));
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colFilter").get())
            ->InsertData(filterNumbers);
        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKey").get())
            ->InsertData(keys, compressedKeyNullMask);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colValue").get())
            ->InsertData(values, compressedValueNullMask);

        // Execute the query
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colKey, " + aggregationFunction +
                                                       "(colValue) FROM " + tableName + " WHERE colFilter > " +
                                                       std::to_string(threshold) + " GROUP BY colKey;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colKey");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colValue)");
        auto& keysNullMaskResult = result->nullbitmasks().at(tableName + ".colKey").nullmask();
        auto& valuesNullMaskResult = result->nullbitmasks().at(aggregationFunction + "(colValue)").nullmask();

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            const char keyChar = keysNullMaskResult[i / 8];
            const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
            const char valueChar = valuesNullMaskResult[i / 8];
            const bool valueIsNull = ((valueChar >> (i % 8)) & 1);
            std::string key = keyIsNull ? "" : payloadKeys.stringpayload().stringdata()[i];
            int32_t value = valueIsNull ? -1 : payloadValues.intpayload().intdata()[i];
            std::cout << key << " " << value << std::endl;
            ASSERT_FALSE(expectedResult.find({key, keyIsNull}) == expectedResult.end())
                << " key " << (keyIsNull ? "NULL" : key) << " not found";

            ASSERT_EQ(std::get<0>(expectedResult.at({key, keyIsNull})), value)
                << " value at key " << (keyIsNull ? "NULL" : key);
            ASSERT_EQ(std::get<1>(expectedResult.at({key, keyIsNull})), valueIsNull)
                << " value null bit at key " << (keyIsNull ? "NULL" : key);
        }
    }

    void GroupByIntNoAgg(std::vector<int32_t> inKeys)
    {
        std::string tableName = "NoAggTable";
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
            ->InsertData(inKeys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colKeys FROM " + tableName + " GROUP BY colKeys;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
        std::set<int32_t> expectedResult;
        for (auto value : inKeys)
        {
            expectedResult.insert(value);
        }
        ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
        {
            int32_t key = payloadKeys.intpayload().intdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
        }
    }

    void GroupByStringNoAgg(std::vector<std::string> inKeys)
    {
        std::string tableName = "NoAggTable";
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_STRING));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
            ->InsertData(inKeys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT colKeys FROM " + tableName + " GROUP BY colKeys;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
        std::set<std::string> expectedResult;
        for (auto value : inKeys)
        {
            expectedResult.insert(value);
        }
        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
        }
    }

    void GroupByStringNoAggAsterisk(std::vector<std::string> inKeys)
    {
        std::string tableName = "NoAggTable";
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_STRING));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
            ->InsertData(inKeys);

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, "SELECT * FROM " + tableName + " GROUP BY colKeys;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
        std::set<std::string> expectedResult;
        for (auto value : inKeys)
        {
            expectedResult.insert(value);
        }
        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
        }
    }

    void GroupByMultiKeyIntIntStringNoAgg(
        std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<std::string>> keys)
    {
        std::string tableName = "NoAggTable";
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colKeysInt1", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKeysInt2", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colKeysString", DataType::COLUMN_STRING));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeysInt1").get())
            ->InsertData(std::get<0>(keys));

        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeysInt2").get())
            ->InsertData(std::get<1>(keys));

        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colKeysString").get())
            ->InsertData(std::get<2>(keys));

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase,
                                  "SELECT colKeysInt1, colKeysInt2, colKeysString FROM " + tableName +
                                      " GROUP BY colKeysInt1, colKeysInt2, colKeysString;");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeysInt1 = result->payloads().at(tableName + ".colKeysInt1");
        auto& payloadKeysInt2 = result->payloads().at(tableName + ".colKeysInt2");
        auto& payloadKeysString = result->payloads().at(tableName + ".colKeysString");

        std::set<std::tuple<int32_t, int32_t, std::string>> expectedResult;

        for (int32_t i = 0; i < std::get<0>(keys).size(); i++)
        {
            expectedResult.insert({std::get<0>(keys)[i], std::get<1>(keys)[i], std::get<2>(keys)[i]});
        }
        ASSERT_EQ(expectedResult.size(), payloadKeysInt1.intpayload().intdata_size())
            << " wrong number of keys";
        ASSERT_EQ(expectedResult.size(), payloadKeysInt2.intpayload().intdata_size())
            << " wrong number of keys";
        ASSERT_EQ(expectedResult.size(), payloadKeysString.stringpayload().stringdata_size())
            << " wrong number of keys";

        std::vector<std::tuple<int32_t, int32_t, std::string>> realResult;

        for (int32_t i = 0; i < payloadKeysInt1.intpayload().intdata_size(); i++)
        {
            int32_t keyInt1 = payloadKeysInt1.intpayload().intdata()[i];
            int32_t keyInt2 = payloadKeysInt2.intpayload().intdata()[i];
            std::string keyString = payloadKeysString.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find({keyInt1, keyInt2, keyString}) == expectedResult.end())
                << " key \"" << keyInt1 << keyInt2 << keyString << "\"";
        }
    }

    template <typename EXCEPTION>
    void GroupByExceptionGeneric(const std::string& tableName, std::vector<std::string> columnNames, std::string query)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        for (auto& column : columnNames)
        {
            columns.insert({column, DataType::COLUMN_INT});
        }
        groupByDatabase->CreateTable(columns, tableName.c_str());

        // Execute the query_
        GpuSqlCustomParser parser(groupByDatabase, query);
        ASSERT_THROW(parser.Parse(), EXCEPTION);
    }
};

// Group By basic numeric keys
TEST_F(DispatcherGroupByTests, IntSimpleSum)
{
    GroupByIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                          {{0, 2}, {1, 6}, {2, 15}, {-1, 4}});
}

TEST_F(DispatcherGroupByTests, IntCollisionsSum)
{
    GroupByIntGenericTest(
        "SUM", {0, 1, -1, 2, 262143, 262144, 262145, 1048576, 1048577, 1, 1, 262144, 0},
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096},
        {{0, 4097}, {1, 1538}, {-1, 4}, {2, 8}, {262143, 16}, {262144, 2080}, {262145, 64}, {1048576, 128}, {1048577, 256}});
}

TEST_F(DispatcherGroupByTests, IntSparseKeysSum)
{
    GroupByIntGenericTest("SUM", {2, 30, 7, 2, 30, 7, 2, 2}, {1, 15, 14, 1, 15, 14, 1, 1},
                          {{2, 4}, {7, 28}, {30, 30}});
}
TEST_F(DispatcherGroupByTests, IntSparseKeysAvg)
{
    GroupByIntGenericTest("AVG", {2, 30, 7, 2, 30, 7, 2, 2}, {1, 15, 14, 1, 15, 14, 1, 1},
                          {{2, 1}, {7, 14}, {30, 15}});
}

TEST_F(DispatcherGroupByTests, IntSparseKeysCount)
{
    GroupByIntCountTest({2, 30, 7, 2, 30, 7, 2, 2}, {1, 15, 0, 1, 15, 0, 1, 1}, {{2, 4}, {7, 2}, {30, 2}});
}

TEST_F(DispatcherGroupByTests, IntSimpleSumNumericAlias)
{
    GroupByIntAliasGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                               {{0, 2}, {1, 6}, {2, 15}, {-1, 4}});
}

TEST_F(DispatcherGroupByTests, IntKeyOpSimpleSum)
{
    GroupByKeyOpIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                               {{2, 2}, {3, 6}, {4, 15}, {1, 4}});
}

TEST_F(DispatcherGroupByTests, IntKeyWhereSimpleSum)
{
    GroupByKeyWhereIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1},
                                  {1, 2, 2, 2, 1, 3, 15, 5, -4}, 1, {{1, 6}});
}

TEST_F(DispatcherGroupByTests, IntKeyOpWhereSimpleSum)
{
    GroupByKeyOpWhereIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1},
                                    {1, 2, 2, 2, 1, 3, 15, 5, -4}, 3, {{1, 6}});
}

TEST_F(DispatcherGroupByTests, IntKeyWhereDelimitedAliasSimpleSum)
{
    GroupByKeyWhereDelimitedAliasIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1},
                                                {1, 2, 2, 2, 1, 3, 15, 5, -4}, 1, {{1, 6}});
}

TEST_F(DispatcherGroupByTests, IntSimpleMin)
{
    GroupByIntGenericTest("MIN", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                          {{0, 1}, {1, -4}, {2, 15}, {-1, 2}});
}

TEST_F(DispatcherGroupByTests, IntSimpleMax)
{
    GroupByIntGenericTest("MAX", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                          {{0, 1}, {1, 5}, {2, 15}, {-1, 2}});
}

TEST_F(DispatcherGroupByTests, IntSimpleAvg)
{
    GroupByIntGenericTest("AVG", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                          {{0, 1}, {1, 1}, {2, 15}, {-1, 2}});
}

TEST_F(DispatcherGroupByTests, IntSimpleCount)
{
    GroupByIntCountTest({0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                        {{0, 2}, {1, 4}, {2, 1}, {-1, 2}});
}

TEST_F(DispatcherGroupByTests, IntSimpleCountAsterisk)
{
    GroupByIntCountAsteriskTest({0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                                {{0, 2}, {1, 4}, {2, 1}, {-1, 2}});
}


// Group By String keys
TEST_F(DispatcherGroupByTests, StringSimpleSum)
{
    GBSGenericTest("SUM", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                   {{"Apple", 4}, {"Abcd", 9}, {"Banana", 5}, {"XYZ", 38}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleMin)
{
    GBSGenericTest("MIN", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                   {{"Apple", 1}, {"Abcd", 2}, {"Banana", 5}, {"XYZ", 4}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleMax)
{
    GBSGenericTest("MAX", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                   {{"Apple", 3}, {"Abcd", 7}, {"Banana", 5}, {"XYZ", 15}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleAvg)
{
    GBSGenericTest("AVG", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                   {{"Apple", 2}, {"Abcd", 4}, {"Banana", 5}, {"XYZ", 9}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleCount)
{
    GBSCountTest({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                 {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                 {{"Apple", 2}, {"Abcd", 2}, {"Banana", 1}, {"XYZ", 4}, {"0", 1}});
}

TEST_F(DispatcherGroupByTests, StringSimpleCountStringWhere)
{
    GBSCountStringWhereTest({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                            {1, 2, 3, 4, 5, 6, 7, 10, 13, 15}, {{"Abcd", 1}, {"XYZ", 3}, {"0", 1}});
}

TEST_F(DispatcherGroupByTests, StringSimpleCountString)
{
    GBSCountStringTest({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                       {{"Apple", 2}, {"Abcd", 2}, {"Banana", 1}, {"XYZ", 4}, {"0", 1}});
}

TEST_F(DispatcherGroupByTests, StringSimpleSumOrderBy)
{
    GBSOBGenericTest("SUM", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                     {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                     {{"Apple", 4}, {"Banana", 5}, {"Abcd", 9}, {"0", 10}, {"XYZ", 38}});
}

TEST_F(DispatcherGroupByTests, StringKeyOpSimpleSum)
{
    GBSKeyOpGenericTest("SUM", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                        {1, 2, 3, 4, 5, 6, 7, 10, 13, 15},
                        {{"App", 4}, {"Abc", 9}, {"Ban", 5}, {"XYZ", 38}, {"0", 10}});
}


// Group By Multi-keys
TEST_F(DispatcherGroupByTests, MultiKeySingleBlockSum)
{
    GroupByMultiKeyGenericTest("SUM", {{1, 1, 1, 2}, {2, 2, 5, 1}}, {5, 7, 24, 1},
                               {{{1, 2}, 12}, {{1, 5}, 24}, {{2, 1}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleSum)
{
    GroupByMultiKeyGenericTest("SUM", {{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}},
                               {5, 5, 24, 1, 7, 1, 1, 2},
                               {{{1, 2}, 10}, {{1, 5}, 24}, {{2, 1}, 1}, {{5, 1}, 9}, {{7, 7}, 1}, {{-1, -5}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleMin)
{
    GroupByMultiKeyGenericTest("MIN", {{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}},
                               {5, 5, 24, 1, 7, 1, 1, 2},
                               {{{1, 2}, 5}, {{1, 5}, 24}, {{2, 1}, 1}, {{5, 1}, 2}, {{7, 7}, 1}, {{-1, -5}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleMax)
{
    GroupByMultiKeyGenericTest("MAX", {{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}},
                               {5, 5, 24, 1, 7, 1, 1, 2},
                               {{{1, 2}, 5}, {{1, 5}, 24}, {{2, 1}, 1}, {{5, 1}, 7}, {{7, 7}, 1}, {{-1, -5}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleAvg)
{
    GroupByMultiKeyGenericTest("AVG", {{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}},
                               {5, 5, 24, 1, 7, 1, 1, 2},
                               {{{1, 2}, 5}, {{1, 5}, 24}, {{2, 1}, 1}, {{5, 1}, 4}, {{7, 7}, 1}, {{-1, -5}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleCount)
{
    GroupByMultiKeyCountTest({{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}},
                             {5, 5, 24, 1, 7, 1, 1, 2},
                             {{{1, 2}, 2}, {{1, 5}, 1}, {{2, 1}, 1}, {{5, 1}, 2}, {{7, 7}, 1}, {{-1, -5}, 1}});
}

TEST_F(DispatcherGroupByTests, MultiKeySimpleCountAsterisk)
{
    GroupByMultiKeyCountAsteriskTest(
        {{1, 1, 1, 2, 5, 7, -1, 5}, {2, 2, 5, 1, 1, 7, -5, 1}}, {5, 5, 24, 1, 7, 1, 1, 2},
        {{{1, 2}, 2}, {{1, 5}, 1}, {{2, 1}, 1}, {{5, 1}, 2}, {{7, 7}, 1}, {{-1, -5}, 1}});
}


TEST_F(DispatcherGroupByTests, MultiKeyIntIntStringSum)
{
    GroupByMultiKeyStringTest("SUM",
                              {{5, 2, 2, 2, 2, 5, 1, 7},
                               {1, 1, 1, 1, 1, 1, 2, 0},
                               {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"}},
                              {5, -3, -3, 9, 7, 5, 4, 20},
                              {{{2, 1, "Apple"}, 9},
                               {{2, 1, "XYZ"}, 7},
                               {{1, 2, "Apple"}, 4},
                               {{7, 0, "Nut"}, 20},
                               {{5, 1, "Apple"}, 10},
                               {{2, 1, "Nut"}, -6}});
}


// Group By + Order By
TEST_F(DispatcherGroupByTests, IntegerSimpleSumOrderByKeys)
{
    GBOBKeysGenericTest("SUM",
                        {
                            10,
                            10,
                            2,
                            2,
                            6,
                            6,
                            4,
                            4,
                            8,
                            8,
                        },
                        {{2, 5}, {4, 13}, {6, 9}, {8, 17}, {10, 1}});
}

TEST_F(DispatcherGroupByTests, IntegerSimpleSumOrderByValues)
{
    GBOBValuesGenericTest("SUM", {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                          {
                              10,
                              10,
                              2,
                              2,
                              6,
                              6,
                              4,
                              4,
                              8,
                              8,
                          },
                          {{10, 1}, {2, 5}, {6, 9}, {4, 13}, {8, 17}});
}


// Group By basic numeric keys
TEST_F(DispatcherGroupByTests, IntSimpleSumValuesOp)
{
    GroupByValueOpIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                                 {{0, -2}, {1, -2}, {2, 13}, {-1, 0}});
}

TEST_F(DispatcherGroupByTests, IntSimpleAggOpOnKey)
{
    GroupByKeyAggOpIntGenericTest("MAX", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {{0, -2}, {1, -1}, {2, 0}, {-1, -3}});
}

TEST_F(DispatcherGroupByTests, NullIntKeysWhere)
{
    GroupByIntWithWhere("SUM", {0, 8, 7, 1, 8, 2, 9, 7}, 4, {1, -1, 1, -1, 2, -1, 2, 5},
                        {false, true, false, true, false, true, false, false},
                        {1, 19, 2, 21, 4, -1, 8, -1}, {false, false, false, false, false, true, false, true},
                        {{{-1, true}, {19, false}},
                         {{1, false}, {2, false}},
                         {{2, false}, {12, false}},
                         {{5, false}, {-1, true}}});
}

TEST_F(DispatcherGroupByTests, NullStringKeysWhere)
{
    GroupByStringWithWhere("SUM", {0, 8, 7, 1, 8, 2, 9, 7}, 4, {"one", "", "one", "", "two", "", "two", "five"},
                           {false, true, false, true, false, true, false, false},
                           {1, 19, 2, 21, 4, -1, 8, -1},
                           {false, false, false, false, false, true, false, true},
                           {{{"", true}, {19, false}},
                            {{"one", false}, {2, false}},
                            {{"two", false}, {12, false}},
                            {{"five", false}, {-1, true}}});
}

TEST_F(DispatcherGroupByTests, IntKeyNoAggSingleBlock)
{
    GroupByIntNoAgg({7, 4, 7, 3});
}

TEST_F(DispatcherGroupByTests, IntKeyNoAggMoreBlocks)
{
    GroupByIntNoAgg({0, 1, 2, 3, 4, 8, -1, 5, -3, -4, -5, -255, 2, 8, -5, 7, 9, 9, 5, 4});
}

TEST_F(DispatcherGroupByTests, StringKeyNoAgg)
{
    GroupByStringNoAgg({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"});
}

TEST_F(DispatcherGroupByTests, StringKeyNoAggAsterisk)
{
    GroupByStringNoAggAsterisk({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"});
}

TEST_F(DispatcherGroupByTests, MultiKeyNoAgg)
{
    GroupByMultiKeyIntIntStringNoAgg({{5, 2, 2, 2, 2, 5, 1, 7},
                                      {1, 1, 1, 1, 1, 1, 2, 0},
                                      {"Apple", "Nut", "Nut", "Apple", "XYZ", "Apple", "Apple", "Nut"}});
}


// Test automatic increase of hash table size
TEST_F(DispatcherGroupByTests, UnlimitedNumberOfKeysNoAgg)
{
    const int32_t buckets =
        Configuration::GetInstance().GetGroupByBuckets() * Context::getInstance().getDeviceCount();
    const int32_t uniqueKeys = buckets * 3 / 2;
    const int32_t inputSize = uniqueKeys * 9 / 8;
    const int32_t mBlockSize = Configuration::GetInstance().GetGroupByBuckets() / 2;
    std::cout << "Test UnlimitedNumberOfKeys: overall_gpus_buckets=" << buckets << ", unique_keys=" << uniqueKeys
              << ", input_size=" << inputSize << ", block_size=" << mBlockSize << std::endl;
    std::vector<int32_t> inKeys(inputSize);
    for (size_t i = 0; i < inputSize; i++)
    {
        inKeys[i] = i % uniqueKeys;
    }

    std::shared_ptr<Database> unlimitedDatabase = std::make_shared<Database>("UnlimitedDb", mBlockSize);

    std::string tableName = "NoAggTable";
    auto columns = std::unordered_map<std::string, DataType>();
    columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_INT));
    unlimitedDatabase->CreateTable(columns, tableName.c_str());

    reinterpret_cast<ColumnBase<int32_t>*>(
        unlimitedDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
        ->InsertData(inKeys);

    // Execute the query_
    GpuSqlCustomParser parser(unlimitedDatabase, "SELECT colKeys FROM " + tableName + " GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
    std::set<int32_t> expectedResult;
    for (auto value : inKeys)
    {
        expectedResult.insert(value);
    }
    ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
        << " wrong number of keys";
    for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
    {
        int32_t key = payloadKeys.intpayload().intdata()[i];
        ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
    }
}

TEST_F(DispatcherGroupByTests, UnlimitedNumberOfKeysInOneThreadNoAgg)
{
    const int32_t buckets =
        Configuration::GetInstance().GetGroupByBuckets() * Context::getInstance().getDeviceCount();
    const int32_t uniqueKeys = buckets * 5 / 2;
    const int32_t inputSize = uniqueKeys;
    const int32_t mBlockSize = Configuration::GetInstance().GetGroupByBuckets() / 2;
    std::cout << "Test UnlimitedNumberOfKeys: overall_gpus_buckets=" << buckets << ", unique_keys=" << uniqueKeys
              << ", input_size=" << inputSize << ", block_size=" << mBlockSize << std::endl;
    std::vector<int32_t> inKeys(inputSize);
    for (size_t i = 0; i < inputSize; i++)
    {
        inKeys[i] = ((i / mBlockSize % Context::getInstance().getDeviceCount()) == 0) ?
                        (i % mBlockSize) :
                        (i % uniqueKeys);
        // std::cout << i << ": in-key " << inKeys[i] << std::endl;
    }

    std::shared_ptr<Database> unlimitedDatabase = std::make_shared<Database>("UnlimitedDb", mBlockSize);

    std::string tableName = "NoAggTable";
    auto columns = std::unordered_map<std::string, DataType>();
    columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_INT));
    unlimitedDatabase->CreateTable(columns, tableName.c_str());

    reinterpret_cast<ColumnBase<int32_t>*>(
        unlimitedDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
        ->InsertData(inKeys);

    // Execute the query_
    GpuSqlCustomParser parser(unlimitedDatabase, "SELECT colKeys FROM " + tableName + " GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
    std::set<int32_t> expectedResult;
    for (auto value : inKeys)
    {
        expectedResult.insert(value);
    }
    ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
        << " wrong number of keys";
    for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
    {
        int32_t key = payloadKeys.intpayload().intdata()[i];
        ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
        // std::cout << i << ": key " << key << std::endl;
    }
}

TEST_F(DispatcherGroupByTests, UnlimitedNumberOfKeysAverage)
{
    const int32_t buckets =
        Configuration::GetInstance().GetGroupByBuckets() * Context::getInstance().getDeviceCount();
    const int32_t uniqueKeys = buckets * 3 / 2;
    const int32_t inputSize = uniqueKeys * 9 / 8;
    const int32_t mBlockSize = Configuration::GetInstance().GetGroupByBuckets() / 2;
    const int32_t CONST_VAL = 7;
    std::cout << "Test UnlimitedNumberOfKeys: overall_gpus_buckets=" << buckets << ", unique_keys=" << uniqueKeys
              << ", input_size=" << inputSize << ", block_size=" << mBlockSize << std::endl;
    std::vector<int32_t> inKeys(inputSize);
    std::vector<int32_t> inValues(inputSize, CONST_VAL);
    for (size_t i = 0; i < inputSize; i++)
    {
        inKeys[i] = i % uniqueKeys;
    }

    std::shared_ptr<Database> unlimitedDatabase = std::make_shared<Database>("UnlimitedDb", mBlockSize);

    std::string tableName = "NoAggTable";
    auto columns = std::unordered_map<std::string, DataType>();
    columns.insert(std::make_pair<std::string, DataType>("colKeys", DataType::COLUMN_INT));
    columns.insert(std::make_pair<std::string, DataType>("colValues", DataType::COLUMN_INT));
    unlimitedDatabase->CreateTable(columns, tableName.c_str());

    reinterpret_cast<ColumnBase<int32_t>*>(
        unlimitedDatabase->GetTables().at(tableName).GetColumns().at("colKeys").get())
        ->InsertData(inKeys);
    reinterpret_cast<ColumnBase<int32_t>*>(
        unlimitedDatabase->GetTables().at(tableName).GetColumns().at("colValues").get())
        ->InsertData(inValues);

    // Execute the query_
    GpuSqlCustomParser parser(unlimitedDatabase,
                              "SELECT colKeys, AVG(colValues) FROM " + tableName + " GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloadKeys = result->payloads().at(tableName + ".colKeys");
    auto& payloadValues = result->payloads().at("AVG(colValues)");
    std::unordered_map<int32_t, int32_t> expectedResult;
    for (auto value : inKeys)
    {
        if (expectedResult.find(value) == expectedResult.end())
        {
            expectedResult.insert({value, CONST_VAL});
        }
    }
    ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
        << " wrong number of keys";
    for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
    {
        const int32_t key = payloadKeys.intpayload().intdata()[i];
        const int32_t val = payloadValues.intpayload().intdata()[i];
        ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key " << key;
        ASSERT_EQ(expectedResult.at(key), val) << " at key " << key;
    }
}

TEST_F(DispatcherGroupByTests, SelectAsteristGroupByException)
{
    GroupByExceptionGeneric<ColumnGroupByException>("TableA", {"colA", "colB", "colC"},
                                                    "SELECT * FROM TableA GROUP BY colA;");
}
