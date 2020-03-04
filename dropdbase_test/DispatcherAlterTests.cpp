#include <boost/functional/hash.hpp>
#include <boost/filesystem.hpp>

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
    const std::string path = Configuration::GetInstance().GetDatabaseDir();
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

        // clear directory to make sure, there are no old database files, but do not remove directory:
        boost::filesystem::path path_to_remove(path);
        for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove); it != end_dir_it; ++it)
        {
            boost::filesystem::remove_all(it->path());
        }
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

        // test in memory changes:
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

        // persist database, rename again, and test changes in memory and on disk
        alterDatabase->Persist();

        std::string newestColName = newColName + "_secondTry";

        GpuSqlCustomParser parser2(alterDatabase, "ALTER TABLE SimpleTable RENAME COLUMN " +
                                                      newColName + " TO " + newestColName + ";");
        resultPtr = parser2.Parse();

        // test in memory changes:
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find(newestColName) !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colA") ==
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find(newColName) ==
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colB") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colC") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());
        ASSERT_TRUE(alterDatabase->GetTables().at("SimpleTable").GetColumns().find("colD") !=
                    alterDatabase->GetTables().at("SimpleTable").GetColumns().end());

        // test changes on disk:
        ASSERT_TRUE(boost::filesystem::exists(boost::filesystem::path(path + dbName + Database::DB_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName + Database::SEPARATOR +
                                    newColName + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName + Database::SEPARATOR +
                                    newestColName + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));

        // clear directory to make sure, there are no old database files, but do not remove directory:
        boost::filesystem::path path_to_remove(path);
        for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove); it != end_dir_it; ++it)
        {
            boost::filesystem::remove_all(it->path());
        }
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

        // test in memory changes:
        ASSERT_TRUE(alterDatabase->GetTables().find("SimpleTable") == alterDatabase->GetTables().end());
        ASSERT_TRUE(alterDatabase->GetTables().find(newTableName) != alterDatabase->GetTables().end());

        // persist database, rename again, and test changes in memory and on disk
        alterDatabase->Persist();
        std::string newestTableName = newTableName + "_secondTry";

        GpuSqlCustomParser parser2(alterDatabase, "ALTER TABLE " + newTableName + " RENAME TO " +
                                                      newestTableName + ";");
        resultPtr = parser2.Parse();

        // test in memory changes:
        ASSERT_TRUE(alterDatabase->GetTables().find(newTableName) == alterDatabase->GetTables().end());
        ASSERT_TRUE(alterDatabase->GetTables().find(newestTableName) != alterDatabase->GetTables().end());

        // test changes on disk:
        ASSERT_TRUE(boost::filesystem::exists(boost::filesystem::path(path + dbName + Database::DB_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newestTableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newestTableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newestTableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newestTableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newTableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newTableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newTableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + newTableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + "SimpleTable" +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + "SimpleTable" +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + "SimpleTable" +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + dbName + Database::SEPARATOR + "SimpleTable" +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));

        // clear directory to make sure, there are no old database files, but do not remove directory:
        boost::filesystem::path path_to_remove(path);
        for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove); it != end_dir_it; ++it)
        {
            boost::filesystem::remove_all(it->path());
        }
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

        // test in memory changes:
        ASSERT_FALSE(Database::Exists("AlterTestDb"));
        ASSERT_TRUE(Database::Exists(newDatabaseName));

        // persist database, rename again, and test changes in memory and on disk
        alterDatabase->Persist();
        std::string newestDatabaseName = newDatabaseName + "_secondTry";

        GpuSqlCustomParser parser2(nullptr, "ALTER DATABASE " + newDatabaseName + " RENAME TO " +
                                                newestDatabaseName + ";");
        resultPtr = parser2.Parse();

        // test in memory changes:
        ASSERT_FALSE(Database::Exists("AlterTestDb"));
        ASSERT_FALSE(Database::Exists(newDatabaseName));
        ASSERT_TRUE(Database::Exists(newestDatabaseName));

        // test changes on disk:
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + newestDatabaseName + Database::DB_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + newDatabaseName + Database::DB_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + "AlterTestDb" + Database::DB_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + newestDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + newestDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + newestDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_TRUE(boost::filesystem::exists(
            boost::filesystem::path(path + newestDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + newDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + newDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + newDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + newDatabaseName + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + "AlterTestDb" + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colA" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + "AlterTestDb" + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colB" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + "AlterTestDb" + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colC" + Database::COLUMN_DATA_EXTENSION)));
        ASSERT_FALSE(boost::filesystem::exists(
            boost::filesystem::path(path + "AlterTestDb" + Database::SEPARATOR + tableName +
                                    Database::SEPARATOR + "colD" + Database::COLUMN_DATA_EXTENSION)));

        // clear directory to make sure, there are no old database files, but do not remove directory:
        boost::filesystem::path path_to_remove(path);
        for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove); it != end_dir_it; ++it)
        {
            boost::filesystem::remove_all(it->path());
        }
    }

    void AlterTableAlterBlockSizeGenericTest(int32_t newBlockSize)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colD", DataType::COLUMN_INT));

        alterDatabase->CreateTable(columns, tableName.c_str());

        ASSERT_NE(blockSize, newBlockSize);

        ASSERT_EQ(alterDatabase->GetTables().at(tableName).GetBlockSize(), blockSize);

        GpuSqlCustomParser parser(alterDatabase, "ALTER TABLE " + tableName + " ALTER BLOCK SIZE " +
                                                     std::to_string(newBlockSize) + ";");
        auto resultPtr = parser.Parse();

        ASSERT_EQ(alterDatabase->GetTables().at(tableName).GetBlockSize(), newBlockSize);
    }

    void AlterDatabaseAlterBlockSizeGenericTest(int32_t newBlockSize)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colA", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colB", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colC", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colD", DataType::COLUMN_INT));

        alterDatabase->CreateTable(columns, tableName.c_str());

        ASSERT_NE(blockSize, newBlockSize);

        ASSERT_EQ(alterDatabase->GetTables().at(tableName).GetBlockSize(), blockSize);
        ASSERT_EQ(alterDatabase->GetBlockSize(), blockSize);

        GpuSqlCustomParser parser(nullptr, "ALTER DATABASE " + dbName + " ALTER BLOCK SIZE " +
                                               std::to_string(newBlockSize) + ";");
        auto resultPtr = parser.Parse();

        ASSERT_EQ(alterDatabase->GetTables().at(tableName).GetBlockSize(), newBlockSize);
        ASSERT_EQ(alterDatabase->GetBlockSize(), newBlockSize);
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

TEST_F(DispatcherAlterTests, AlterTableBlockSizeTest)
{
    AlterTableAlterBlockSizeGenericTest(blockSize * 2);
}

TEST_F(DispatcherAlterTests, AlterDatabaseBlockSizeTest)
{
    AlterDatabaseAlterBlockSizeGenericTest(blockSize * 2);
}
