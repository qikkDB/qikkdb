#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"

class ShowTests : public ::testing::Test
{
protected:
    const std::string path = Configuration::GetInstance().GetDatabaseDir();
    const std::string dbName = "ShowDb";
    const std::string tableName = "tableA";
    const int32_t blockSize = 15; // length of a block

    std::shared_ptr<Database> showDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        showDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(showDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    void ShowConstraints()
    {
        GpuSqlCustomParser parser(showDatabase, "CREATE TABLE " + tableName + " (colA int NOT NULL, colB string, NOT NULL n(colB), UNIQUE u(colA, colB));");
        auto resultPtr = parser.Parse();

        GpuSqlCustomParser parserShow(showDatabase, "SHOW CONSTRAINTS FROM " + tableName + ";");
        auto resultPtrShow = parserShow.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtrShow.get());

        std::vector<std::string> expectedConstraintsNames;
        expectedConstraintsNames.push_back("colA_NC");
        expectedConstraintsNames.push_back("n");
        expectedConstraintsNames.push_back("u");

        std::vector<std::string> expectedConstraintsTypes;
        expectedConstraintsTypes.push_back("NOT NULL");
        expectedConstraintsTypes.push_back("NOT NULL");
        expectedConstraintsTypes.push_back("UNIQUE");

        std::vector<std::string> expectedConstraintsColumns;
        expectedConstraintsColumns.push_back("colA");
        expectedConstraintsColumns.push_back("colB");
        expectedConstraintsColumns.push_back("colA\ncolB");

        auto& payloadsConstraintsNames = result->payloads().at(tableName + "_constraints");
        auto& payloadsConstraintsTypes = result->payloads().at(tableName + "_cnstrn_types");
        auto& payloadsConstraintsColumns = result->payloads().at(tableName + "_cnstrn_cols");

        ASSERT_EQ(expectedConstraintsNames.size(), payloadsConstraintsNames.stringpayload().stringdata_size());
        ASSERT_EQ(expectedConstraintsTypes.size(), payloadsConstraintsTypes.stringpayload().stringdata_size());
        ASSERT_EQ(expectedConstraintsColumns.size(),
                  payloadsConstraintsColumns.stringpayload().stringdata_size());

        for (int i = 0; i < expectedConstraintsNames.size(); i++)
        {
            std::vector<std::string>::iterator it =
                std::find(expectedConstraintsNames.begin(), expectedConstraintsNames.end(),
                          payloadsConstraintsNames.stringpayload().stringdata()[i]);

            ASSERT_TRUE(it != expectedConstraintsNames.end());

            int index = std::distance(expectedConstraintsNames.begin(), it);

            ASSERT_EQ(expectedConstraintsTypes[index],
                      payloadsConstraintsTypes.stringpayload().stringdata()[i]);
            ASSERT_EQ(expectedConstraintsColumns[index],
                      payloadsConstraintsColumns.stringpayload().stringdata()[i]);
        }
    }
};

TEST_F(ShowTests, showConstraints)
{
    ShowConstraints();
}