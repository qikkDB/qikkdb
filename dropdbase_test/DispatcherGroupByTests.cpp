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

        //ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
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
};

// Group By basic numeric keys
TEST_F(DispatcherGroupByTests, IntSimpleSum)
{
    GroupByIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                          {{0, 2}, {1, 6}, {2, 15}, {-1, 4}});
}

TEST_F(DispatcherGroupByTests, IntCollisionsSum)
{
    GroupByIntGenericTest("SUM", {0, 1, -1, 2, 262143, 262144, 262145, 1048576, 1048577, 1, 1, 262144, 0}, {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096},
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
    GroupByIntCountTest({2, 30, 7, 2, 30, 7, 2, 2}, {1, 15, 0, 1, 15, 0, 1, 1},
                          {{2, 4}, {7, 2}, {30, 2}});
}

TEST_F(DispatcherGroupByTests, IntKeyOpSimpleSum)
{
    GroupByKeyOpIntGenericTest("SUM", {0, 1, -1, -1, 0, 1, 2, 1, 1}, {1, 2, 2, 2, 1, 3, 15, 5, -4},
                               {{2, 2}, {3, 6}, {4, 15}, {1, 4}});
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
    GroupByMultiKeyGenericTest("SUM", {{1, 1, 1, 2}, {2, 2, 5, 1}},
                               {5, 7, 24, 1},
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
        GroupByKeyAggOpIntGenericTest("MAX",
                                      {0, 1, -1, -1, 0, 1, 2, 1, 1},
                                      {{0, -2},{1, -1},{2, 0},{-1, -3}});
}
