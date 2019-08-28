#include <boost/functional/hash.hpp>

#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

class DispatcherCastTests : public ::testing::Test
{
protected:
    const std::string dbName = "CastTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> castDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        castDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(castDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    void CastStringToIntGenericTest(std::vector<std::string> strings, std::vector<int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        castDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(strings);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colString AS INT) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colStringASINT)");

        ASSERT_EQ(expectedResult.size(), payloadCast.intpayload().intdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.intpayload().intdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payloadCast.intpayload().intdata()[i]);
        }
    }

    void CastStringToFloatGenericTest(std::vector<std::string> strings, std::vector<float> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        castDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(strings);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colString AS FLOAT) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colStringASFLOAT)");

        ASSERT_EQ(expectedResult.size(), payloadCast.floatpayload().floatdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.floatpayload().floatdata_size(); i++)
        {
            ASSERT_FLOAT_EQ(expectedResult[i], payloadCast.floatpayload().floatdata()[i]) << i;
        }
    }
};

TEST_F(DispatcherCastTests, StringToIntTest)
{
    CastStringToIntGenericTest({"2", "20.7", "30", "40", "123.1", "123123", "0", "-20"},
                               {2, 20, 30, 40, 123, 123123, 0, -20});
}

TEST_F(DispatcherCastTests, StringToFloatTest)
{
    CastStringToFloatGenericTest({"2.0", "20.5", "30", "40.34", "123.78", "123123.4", "0.2", "-20.01"},
                                 {2.0f, 20.5f, 30.0f, 40.34f, 123.78f, 123123.4f, 0.2f, -20.01f});
}

TEST_F(DispatcherCastTests, StringToFloatExpNotationTest)
{
    CastStringToFloatGenericTest({"+1e2", "1.24e3", "1e1", "1e0", "1e-2", "-10.24e-1", "1e-1", "-1e0"},
                                 {100.0f, 1240.0f, 10.0f, 1.0f, 0.01f, -1.024f, 0.1f, -1.0f});
}