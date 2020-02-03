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

    void ShowQueryColumnTypes()
    {
        GpuSqlCustomParser parser(showDatabase, "CREATE TABLE " + tableName + " (colA int, colB string, colC double);");
        auto resultPtr = parser.Parse();
        GpuSqlCustomParser parserShow(showDatabase,
                                      "SHOW QUERY COLUMN TYPES select colB, colC from " + tableName + ";");
        auto resultPtrShow = parserShow.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtrShow.get());

        auto& payloadsColumnNames = result->payloads().at("ColumnName");
        auto& payloadsColumnTypes = result->payloads().at("TypeName");

        std::vector<std::string> expectedColumnNames;
        expectedColumnNames.push_back("tableA.colB");
        expectedColumnNames.push_back("tableA.colC");

        std::vector<std::string> expectedColumnTypes;
        expectedColumnTypes.push_back(GetStringFromColumnDataType(DataType::COLUMN_STRING));
        expectedColumnTypes.push_back(GetStringFromColumnDataType(DataType::COLUMN_DOUBLE));

        ASSERT_EQ(expectedColumnNames.size(), payloadsColumnNames.stringpayload().stringdata_size());
        ASSERT_EQ(expectedColumnTypes.size(), payloadsColumnTypes.stringpayload().stringdata_size());

        for (int i = 0; i < expectedColumnNames.size(); i++)
        {
            ASSERT_EQ(expectedColumnTypes[i], payloadsColumnTypes.stringpayload().stringdata()[i]);
            ASSERT_EQ(expectedColumnNames[i], payloadsColumnNames.stringpayload().stringdata()[i]);
        }
    }
};

TEST_F(ShowTests, showConstraints)
{
    ShowConstraints();
}

TEST_F(ShowTests, showQueryColumnTypes)
{
    ShowQueryColumnTypes();
}