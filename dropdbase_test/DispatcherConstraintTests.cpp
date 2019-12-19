#include <cmath>

#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/BlockBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/GpuSqlParser/ParserExceptions.h"
#include "DispatcherObjs.h"

TEST(DispatcherConstraintTests, UniqueTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, UNIQUE colAUnique (colA), "
                              "NOT NULL colANotNull(colA));");
    auto resultPtr = parser.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colA") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());
    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colA")
                    ->GetIsUnique());

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");
    resultPtr = parser2.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, NotNullTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT NOT NULL);");
    auto resultPtr = parser.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colA") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());
    ASSERT_FALSE(DispatcherObjs::GetInstance()
                     .database->GetTables()
                     .at("TableConstraint")
                     .GetColumns()
                     .at("colA")
                     ->GetIsNullable());

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");
    resultPtr = parser2.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, UniqueGroupTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, UNIQUE u "
                              "(colA, colB), NOT NULL n (colA, colB));");
    auto resultPtr = parser.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colA") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colB") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());

    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colA")
                    ->GetIsUnique());

    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colB")
                    ->GetIsUnique());

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");
    resultPtr = parser2.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, NotNullGroupTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, NOT NULL n "
                              "(colA, colB));");
    auto resultPtr = parser.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colA") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());
    ASSERT_TRUE(
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().find("colB") !=
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetColumns().end());

    ASSERT_FALSE(DispatcherObjs::GetInstance()
                     .database->GetTables()
                     .at("TableConstraint")
                     .GetColumns()
                     .at("colA")
                     ->GetIsNullable());

    ASSERT_FALSE(DispatcherObjs::GetInstance()
                     .database->GetTables()
                     .at("TableConstraint")
                     .GetColumns()
                     .at("colB")
                     ->GetIsNullable());

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");
    resultPtr = parser2.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, SetBlockSizePerTableTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableBlockSizeA (colA INT);");
    auto resultPtr = parser.Parse();

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "CREATE TABLE TableBlockSizeB 10 (colA INT);");
    resultPtr = parser2.Parse();

    GpuSqlCustomParser parser3(DispatcherObjs::GetInstance().database,
                               "CREATE TABLETableBlockSizeC 20 (colA INT);");
    resultPtr = parser3.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeA") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeB") !=
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeC") !=
                DispatcherObjs::GetInstance().database->GetTables().end());

    ASSERT_EQ(DispatcherObjs::GetInstance().database->GetTables().at("TableBlockSizeA").GetBlockSize(),
              DispatcherObjs::GetInstance().database->GetBlockSize());

    ASSERT_EQ(DispatcherObjs::GetInstance().database->GetTables().at("TableBlockSizeB").GetBlockSize(), 10);

    ASSERT_EQ(DispatcherObjs::GetInstance().database->GetTables().at("TableBlockSizeC").GetBlockSize(), 20);

    GpuSqlCustomParser parser4(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableBlockSizeA;");
    resultPtr = parser4.Parse();

    GpuSqlCustomParser parser5(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableBlockSizeB;");
    resultPtr = parser5.Parse();

    GpuSqlCustomParser parser6(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableBlockSizeC;");
    resultPtr = parser6.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeA") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeB") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableBlockSizeC") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, CreateTableColumnAlreadyConstrained)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, colC FLOAT, "
                              "NOT NULL n1 (colA, colB), NOT NULL n2 (colA, colC));");

    ASSERT_THROW(parser.Parse(), ColumnAlreadyConstrainedException);
}

TEST(DispatcherConstraintTests, CreateTableConstraintAlreadyExists)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, colC FLOAT, "
                              "NOT NULL n1 (colA, colB), NOT NULL n1 (colC));");

    ASSERT_THROW(parser.Parse(), ConstraintAlreadyExistsException);
}

TEST(DispatcherConstraintTests, AlterTableColumnAlreadyConstrained)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, colC FLOAT);");
    auto resultPtr = parser.Parse();

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "ALTER TABLE TableConstraint ADD NOT NULL n1 (colA, colB), ADD NOT "
                               "NULL n2 (colA, colC);");

    ASSERT_THROW(parser2.Parse(), ColumnAlreadyConstrainedException);

    GpuSqlCustomParser parser3(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");
    resultPtr = parser3.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, AlterTableConstraintAlreadyExists)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB STRING, colC FLOAT);");
    auto resultPtr = parser.Parse();

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "ALTER TABLE TableConstraint ADD NOT NULL n1 (colA, colB), ADD NOT "
                               "NULL n1 (colC);");

    ASSERT_THROW(parser2.Parse(), ConstraintAlreadyExistsException);

    GpuSqlCustomParser parser3(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");

    resultPtr = parser3.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}

TEST(DispatcherConstraintTests, CreateTableAddDropMultipleConstrainTest)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE TableConstraint (colA INT, colB FLOAT, colC LONG, colD "
                              "DOUBLE INDEX, NOT NULL n (colA, colB), UNIQUE u (colB));");
    auto resultPtr = parser.Parse();

    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") !=
                DispatcherObjs::GetInstance().database->GetTables().end());

    auto& tableConstraints =
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetConstraints();

    ASSERT_EQ(tableConstraints.at("n").first, ConstraintType::CONSTRAINT_NOT_NULL);
    std::vector<std::string> expectedConstraintColumns = {"colA", "colB"};

    for (int32_t i = 0; i < expectedConstraintColumns.size(); i++)
    {
        ASSERT_EQ(tableConstraints.at("n").second[i], expectedConstraintColumns[i]);
    }

    ASSERT_FALSE(DispatcherObjs::GetInstance()
                     .database->GetTables()
                     .at("TableConstraint")
                     .GetColumns()
                     .at("colA")
                     ->GetIsNullable());

    ASSERT_FALSE(DispatcherObjs::GetInstance()
                     .database->GetTables()
                     .at("TableConstraint")
                     .GetColumns()
                     .at("colB")
                     ->GetIsNullable());

    ASSERT_EQ(tableConstraints.at("u").first, ConstraintType::CONSTRAINT_UNIQUE);
    expectedConstraintColumns = {"colB"};

    for (int32_t i = 0; i < expectedConstraintColumns.size(); i++)
    {
        ASSERT_EQ(tableConstraints.at("u").second[i], expectedConstraintColumns[i]);
    }

    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colB")
                    ->GetIsUnique());

    ASSERT_EQ(tableConstraints.at("colD_IC").first, ConstraintType::CONSTRAINT_INDEX);
    expectedConstraintColumns = {"colD"};

    for (int32_t i = 0; i < expectedConstraintColumns.size(); i++)
    {
        ASSERT_EQ(tableConstraints.at("colD_IC").second[i], expectedConstraintColumns[i]);
    }

    auto& sortingColumns =
        DispatcherObjs::GetInstance().database->GetTables().at("TableConstraint").GetSortingColumns();

    ASSERT_TRUE(std::find(sortingColumns.begin(), sortingColumns.end(), "colD") != sortingColumns.end());

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "ALTER TABLE TableConstraint DROP NOT NULL n;");
    resultPtr = parser2.Parse();

    ASSERT_TRUE(tableConstraints.find("n") == tableConstraints.end());

    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colA")
                    ->GetIsNullable());

    ASSERT_TRUE(DispatcherObjs::GetInstance()
                    .database->GetTables()
                    .at("TableConstraint")
                    .GetColumns()
                    .at("colB")
                    ->GetIsNullable());

    GpuSqlCustomParser parser3(DispatcherObjs::GetInstance().database,
                               "DROP TABLE TableConstraint;");

    resultPtr = parser3.Parse();
    ASSERT_TRUE(DispatcherObjs::GetInstance().database->GetTables().find("TableConstraint") ==
                DispatcherObjs::GetInstance().database->GetTables().end());
}
