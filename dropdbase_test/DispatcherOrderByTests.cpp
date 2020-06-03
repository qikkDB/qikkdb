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

    void OrderByNullValues(std::vector<int32_t> colA,
                           std::vector<bool> colANull,
                           bool orderByA,
                           std::vector<int32_t> colB,
                           std::vector<bool> colBNull,
                           bool orderByB,
                           std::vector<int32_t> correctA,
                           std::vector<bool> correctANull,
                           std::vector<int32_t> correctB,
                           std::vector<bool> correctBNull)
    {
        if (colA.size() != colB.size() || colA.size() != colANull.size() || colB.size() != colBNull.size())
        {
            FAIL() << "Input sizes mis-match in test";
        }
        if (correctA.size() != correctB.size() || correctA.size() != correctANull.size() ||
            correctB.size() != correctBNull.size())
        {
            FAIL() << "Correct sizes mis-match in test";
        }
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        orderByDatabase->CreateTable(columns, tableName.c_str());

        for (int row = 0; row < colA.size(); row++)
        {
            GpuSqlCustomParser(orderByDatabase,
                               "INSERT INTO " + tableName + " (colA, colB) VALUES (" +
                                   (colANull[row] ? "NULL" : std::to_string(colA[row])) + ", " +
                                   (colBNull[row] ? "NULL" : std::to_string(colB[row])) + ");")
                .Parse();
        }

        GpuSqlCustomParser parser(orderByDatabase, "SELECT colA, colB FROM " + tableName +
                                                       " ORDER BY " + (orderByA ? "colA" : "") +
                                                       ((orderByA && orderByB) ? ", " : "") +
                                                       (orderByB ? "colB" : "") + ";");
        auto resultPtr = parser.Parse();

        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadA = result->payloads().at(tableName + ".colA");
        auto& payloadANullBitMask = result->nullbitmasks().at(tableName + ".colA").nullmask();
        auto& payloadB = result->payloads().at(tableName + ".colB");
        auto& payloadBNullBitMask = result->nullbitmasks().at(tableName + ".colB").nullmask();

        ASSERT_EQ(correctA.size(), payloadA.intpayload().intdata_size())
            << " wrong number of results";
        ASSERT_EQ(NullValues::GetNullBitMaskSize(correctANull.size()), payloadANullBitMask.size())
            << " wrong number of results (nullMask)";
        ASSERT_EQ(correctB.size(), payloadB.intpayload().intdata_size())
            << " wrong number of results";
        ASSERT_EQ(NullValues::GetNullBitMaskSize(correctBNull.size()), payloadBNullBitMask.size())
            << " wrong number of results (nullMask)";

        for (int32_t i = 0; i < correctA.size(); i++)
        {
            int8_t nullABit = NullValues::GetConcreteBitFromBitmask(payloadANullBitMask.begin(), i);
            int8_t nullBBit = NullValues::GetConcreteBitFromBitmask(payloadBNullBitMask.begin(), i);
            ASSERT_EQ(correctANull[i], nullABit == 1);
            ASSERT_EQ(correctBNull[i], nullBBit == 1);
            if (!nullABit)
            {
                ASSERT_EQ(correctA[i], payloadA.intpayload().intdata()[i]);
            }
            if (!nullABit)
            {
                ASSERT_EQ(correctB[i], payloadB.intpayload().intdata()[i]);
            }
        }
    }

    void OrderByGroupByNullValues(std::vector<int32_t> colA,
                                  std::vector<bool> colANull,
                                  std::vector<int32_t> correctA,
                                  std::vector<bool> correctANull,
                                  bool desc)
    {
        if (colA.size() != colANull.size())
        {
            FAIL() << "Input sizes mis-match in test";
        }
        if (correctA.size() != correctANull.size())
        {
            FAIL() << "Correct sizes mis-match in test";
        }
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        orderByDatabase->CreateTable(columns, tableName.c_str());

        for (int row = 0; row < colA.size(); row++)
        {
            GpuSqlCustomParser(orderByDatabase, "INSERT INTO " + tableName + " (colA) VALUES (" +
                                                    (colANull[row] ? "NULL" : std::to_string(colA[row])) + ");")
                .Parse();
        }

        GpuSqlCustomParser parser(orderByDatabase, "SELECT colA FROM " + tableName + " GROUP BY colA ORDER BY colA" +
                                                       (desc ? " DESC" : "") + ";");
        auto resultPtr = parser.Parse();

        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadA = result->payloads().at(tableName + ".colA");
        auto& payloadANullBitMask = result->nullbitmasks().at(tableName + ".colA").nullmask();

        ASSERT_EQ(correctA.size(), payloadA.intpayload().intdata_size())
            << " wrong number of results";
        ASSERT_EQ(NullValues::GetNullBitMaskSize(correctANull.size()), payloadANullBitMask.size())
            << " wrong number of results (nullMask)";

        for (int32_t i = 0; i < correctA.size(); i++)
        {
            int8_t nullABit = NullValues::GetConcreteBitFromBitmask(payloadANullBitMask.begin(), i);
            ASSERT_EQ(correctANull[i], nullABit == 1);
            if (!nullABit)
            {
                ASSERT_EQ(correctA[i], payloadA.intpayload().intdata()[i]);
            }
        }
    }

    void OrderByGroupByNullValues(std::vector<int32_t> colA,
                                  std::vector<bool> colANull,
                                  std::vector<int32_t> colB,
                                  std::vector<bool> colBNull,
                                  std::vector<int32_t> correctA,
                                  std::vector<bool> correctANull,
                                  std::vector<int32_t> correctB,
                                  std::vector<bool> correctBNull)
    {
        if (colA.size() != colB.size() || colA.size() != colANull.size() || colB.size() != colBNull.size())
        {
            FAIL() << "Input sizes mis-match in test";
        }
        if (correctA.size() != correctB.size() || correctA.size() != correctANull.size() ||
            correctB.size() != correctBNull.size())
        {
            FAIL() << "Correct sizes mis-match in test";
        }
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        orderByDatabase->CreateTable(columns, tableName.c_str());

        for (int row = 0; row < colA.size(); row++)
        {
            GpuSqlCustomParser(orderByDatabase,
                               "INSERT INTO " + tableName + " (colA, colB) VALUES (" +
                                   (colANull[row] ? "NULL" : std::to_string(colA[row])) + ", " +
                                   (colBNull[row] ? "NULL" : std::to_string(colB[row])) + ");")
                .Parse();
        }

        GpuSqlCustomParser parser(orderByDatabase, "SELECT colA, colB FROM " + tableName +
                                                       " GROUP BY colA, colB ORDER BY colA, colB;");
        auto resultPtr = parser.Parse();

        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadA = result->payloads().at(tableName + ".colA");
        auto& payloadANullBitMask = result->nullbitmasks().at(tableName + ".colA").nullmask();
        auto& payloadB = result->payloads().at(tableName + ".colB");
        auto& payloadBNullBitMask = result->nullbitmasks().at(tableName + ".colB").nullmask();

        ASSERT_EQ(correctA.size(), payloadA.intpayload().intdata_size())
            << " wrong number of results";
        ASSERT_EQ(NullValues::GetNullBitMaskSize(correctANull.size()), payloadANullBitMask.size())
            << " wrong number of results (nullMask)";
        ASSERT_EQ(correctB.size(), payloadB.intpayload().intdata_size())
            << " wrong number of results";
        ASSERT_EQ(NullValues::GetNullBitMaskSize(correctBNull.size()), payloadBNullBitMask.size())
            << " wrong number of results (nullMask)";

        for (int32_t i = 0; i < correctA.size(); i++)
        {
            int8_t nullABit = NullValues::GetConcreteBitFromBitmask(payloadANullBitMask.begin(), i);
            int8_t nullBBit = NullValues::GetConcreteBitFromBitmask(payloadBNullBitMask.begin(), i);
            ASSERT_EQ(correctANull[i], nullABit == 1);
            ASSERT_EQ(correctBNull[i], nullBBit == 1);
            if (!nullABit)
            {
                ASSERT_EQ(correctA[i], payloadA.intpayload().intdata()[i]);
            }
            if (!nullABit)
            {
                ASSERT_EQ(correctB[i], payloadB.intpayload().intdata()[i]);
            }
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


TEST_F(DispatcherOrderByTests, OrderByNullValues)
{
    OrderByNullValues({0, 0, 3, 2, 1, 0, 1, 1}, {true, true, false, false, false, false, true, false},
                      true, {5, 6, 4, 5, 5, 7, 0, -1},
                      {false, false, false, false, false, false, false, false}, false,
                      {0, 0, 0, 0, 1, 1, 2, 3}, {true, true, true, false, false, false, false, false},
                      {5, 6, 0, 7, 5, -1, 5, 4}, {false, false, false, false, false, false, false, false});
}

TEST_F(DispatcherOrderByTests, OrderByReorderNullValues)
{
    OrderByNullValues({0, 0, 3, 2, 1, 0, 1, 1}, {true, true, false, false, false, false, true, false},
                      false, {5, 6, 4, 5, 5, 7, 0, -1},
                      {false, false, false, false, false, false, false, false}, true,
                      {1, -1, 3, -1, 2, 1, -1, 0}, {false, true, false, true, false, false, true, false},
                      {-1, 0, 4, 5, 5, 5, 6, 7}, {false, false, false, false, false, false, false, false});
}


TEST_F(DispatcherOrderByTests, OrderByGroupByNullValuesAsc)
{
    OrderByGroupByNullValues({0, 0, -1, 1, 0, -1, -1, 2, -1, -1},
                             {false, false, true, false, false, true, true, false, true, true},
                             {-1, 0, 1, 2}, {true, false, false, false}, false);
}

TEST_F(DispatcherOrderByTests, OrderByGroupByNullValuesDesc)
{
    OrderByGroupByNullValues({0, 0, -1, 1, 0, -1, -1, 2, -1, -1},
                             {false, false, true, false, false, true, true, false, true, true},
                             {2, 1, 0, -1}, {false, false, false, true}, true);
}
