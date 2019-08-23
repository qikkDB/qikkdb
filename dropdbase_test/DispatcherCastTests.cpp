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
};

// Group By basic numeric keys
TEST_F(DispatcherCastTests, StringToIntTest)
{
    CastStringToIntGenericTest({"2", "20", "30", "40", "123", "123123", "0", "-20"},
                               {2, 20, 30, 40, 123, 123123, 0, -20});
}