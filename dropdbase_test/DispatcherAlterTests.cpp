#include <boost/functional/hash.hpp>

#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

class DispatcherAlterTests : public ::testing::Test
{
protected:
    const std::string dbName = "AlterTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> alterDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        alterDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(alterDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    void AlterTableRenameColumnGenericTest(std::string newColName)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colD", DataType::COLUMN_INT));

        alterDatabase->CreateTable(columns, tableName.c_str());

        GpuSqlCustomParser parser(alterDatabase,
                                  "ALTER TABLE SimpleTable RENAME COLUMN colA TO " + newColName + ";");
        auto resultPtr = parser.Parse();


        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find(newColName) !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colA") ==
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colB") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colC") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colD") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
    }

    void AlterTableRenameTableGenericTest(std::string newTableName)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colD", DataType::COLUMN_INT));

        alterDatabase->CreateTable(columns, tableName.c_str());

        GpuSqlCustomParser parser(alterDatabase, "ALTER TABLE SimpleTable RENAME TO " + newTableName + ";");
        auto resultPtr = parser.Parse();

        ASSERT_TRUE(alterDatabase->GetTables().find("SimpleTable") == alterDatabase->GetTables().end());
        ASSERT_TRUE(alterDatabase->GetTables().find(newTableName) != alterDatabase->GetTables().end());
    }

    void AlterTableRenameDatabaseGenericTest(std::string newDatabaseName)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colD", DataType::COLUMN_INT));

        alterDatabase->CreateTable(columns, tableName.c_str());

        GpuSqlCustomParser parser(nullptr, "ALTER DATABASE AlterTestDb RENAME TO " + newDatabaseName + ";");
        auto resultPtr = parser.Parse();

        ASSERT_TRUE(!Database::Exists("AlterTestDb"));
        ASSERT_TRUE(Database::Exists(newDatabaseName));

        GpuSqlCustomParser parser2(nullptr, "ALTER DATABASE " + newDatabaseName + " RENAME TO AlterTestDb;");
        resultPtr = parser2.Parse();

        ASSERT_TRUE(Database::Exists("AlterTestDb"));
        ASSERT_TRUE(!Database::Exists(newDatabaseName));
    }
};

TEST_F(DispatcherAlterTests, RenameColumnTest)
{
    AlterTableRenameColumnGenericTest("colInteger1");
}

TEST_F(DispatcherAlterTests, RenameTableTest)
{
    AlterTableRenameTableGenericTest("TableA");
}

TEST_F(DispatcherAlterTests, RenameDatabaseTest)
{
    AlterTableRenameDatabaseGenericTest("RenamedDatabase");
}
