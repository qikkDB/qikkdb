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

TEST(DispatcherTestsRegression, EmptyResultFromGtColConst)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1 FROM TableA WHERE colInteger1 > 4096;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    ASSERT_EQ(result->payloads().size(), 0); // Check if the result size is also 0
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByCount)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096 "
                              "GROUP BY colInteger1;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupByAvg)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT AVG(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP "
                              "BY colInteger1;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptyResultFromGroupBySum)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096 GROUP "
                              "BY colInteger1;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    ASSERT_EQ(result->payloads().size(), 0);
}


TEST(DispatcherTestsRegression, EmptySetAggregationCount)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT COUNT(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, EmptySetAggregationSum)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT SUM(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_EQ(result->payloads().size(), 0);
    // TODO: assert at row 0
}

TEST(DispatcherTestsRegression, EmptySetAggregationMin)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT MIN(colInteger1) FROM TableA WHERE colInteger1 > 4096;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_EQ(result->payloads().size(), 0);
}

TEST(DispatcherTestsRegression, PointAggregationCount)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT COUNT(colPoint1) FROM TableA;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloads = result->payloads().at("COUNT(colPoint1)");

    ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
    ASSERT_EQ(payloads.int64payload().int64data()[0], TEST_BLOCK_COUNT * TEST_BLOCK_SIZE);
}

TEST(DispatcherTestsRegression, PointAggregationCountWithWhere)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT COUNT(colPoint1) FROM TableA WHERE colInteger1 > 0;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloads = result->payloads().at("COUNT(colPoint1)");

    ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
    // Count sufficient row on CPU
    auto columnInt = dynamic_cast<ColumnBase<int32_t>*>(DispatcherObjs::GetInstance()
                                                            .database->GetTables()
                                                            .at("TableA")
                                                            .GetColumns()
                                                            .at("colInteger1")
                                                            .get());
    int32_t expectedCount = 0;
    for (int i = 0; i < TEST_BLOCK_COUNT; i++)
    {
        auto blockInt = columnInt->GetBlocksList()[i];
        for (int k = 0; k < TEST_BLOCK_SIZE; k++)
        {
            if (blockInt->GetData()[k] > 0)
            {
                expectedCount++;
            }
        }
    }

    ASSERT_EQ(payloads.int64payload().int64data()[0], expectedCount);
}

TEST(DispatcherTestsRegression, Int32AggregationCount)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT COUNT(colInteger1) FROM TableA;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloads = result->payloads().at("COUNT(colInteger1)");

    ASSERT_EQ(payloads.int64payload().int64data_size(), 1);
    ASSERT_EQ(payloads.int64payload().int64data()[0], TEST_BLOCK_COUNT * TEST_BLOCK_SIZE);
}

TEST(DispatcherTestsRegression, GroupByKeyOpCorrectSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT (colInteger1 + 2) * 10, COUNT(colFloat1) FROM TableA GROUP "
                              "BY colInteger1 + 2;");
    ASSERT_NO_THROW(parser.Parse());
}


TEST(DispatcherTestsRegression, GroupByKeyOpWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT (10 * colInteger1) + 2, COUNT(colFloat1) FROM TableA GROUP "
                              "BY colInteger1 + 2;");
    ASSERT_THROW(parser.Parse(), ColumnGroupByException);

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "SELECT colInteger1 + 3, COUNT(colFloat1) FROM TableA GROUP BY "
                               "colInteger1 + 2;");
    ASSERT_THROW(parser2.Parse(), ColumnGroupByException);
}

TEST(DispatcherTestsRegression, NonGroupByAggWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1, SUM(colInteger2) FROM TableA;");
    ASSERT_THROW(parser.Parse(), ColumnGroupByException);
}

TEST(DispatcherTestsRegression, NonGroupByAggCorrectSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT MIN(colInteger1), SUM(colInteger2) FROM TableA;");
    parser.Parse();
}

TEST(DispatcherTestsRegression, AggInWhereWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1, SUM(colInteger2) FROM TableA WHERE "
                              "SUM(colInteger2) > 2;");
    ASSERT_THROW(parser.Parse(), AggregationWhereException);
}

TEST(DispatcherTestsRegression, AggInGroupByWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT SUM(colInteger2) FROM TableA GROUP BY SUM(colInteger2);");
    ASSERT_THROW(parser.Parse(), AggregationGroupByException);
}

TEST(DispatcherTestsRegression, AggInGroupByAliasWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT SUM(colInteger2) FROM TableA GROUP BY 1;");
    ASSERT_THROW(parser.Parse(), AggregationGroupByException);
}

TEST(DispatcherTestsRegression, GroupByAliasDataTypeWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1, SUM(colInteger2) FROM TableA GROUP BY 1.1;");
    ASSERT_THROW(parser.Parse(), GroupByInvalidColumnException);
}

TEST(DispatcherTestsRegression, GroupByAliasOutOfRangeWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1, SUM(colInteger2) FROM TableA GROUP BY 100;");
    ASSERT_THROW(parser.Parse(), GroupByInvalidColumnException);
}

TEST(DispatcherTestsRegression, OrderByAliasDataTypeWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1 FROM TableA ORDER BY 1.1;");
    ASSERT_THROW(parser.Parse(), OrderByInvalidColumnException);
}

TEST(DispatcherTestsRegression, OrderByAliasOutOfRangeWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1 FROM TableA ORDER BY 100;");
    ASSERT_THROW(parser.Parse(), OrderByInvalidColumnException);
}

TEST(DispatcherTestsRegression, ConstOpOnMultiGPU)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database, "SELECT 2+2 FROM TableA;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloads = result->payloads().at("2+2");
    ASSERT_EQ(payloads.intpayload().intdata_size(), 2);
    ASSERT_EQ(payloads.intpayload().intdata()[0], 4);
}

TEST(DispatcherTestsRegression, SameAliasAsColumn)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1 as colInteger1 FROM TableA WHERE colInteger1 > "
                              "20;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
}

// == JOIN ==
TEST(DispatcherTestsRegression, JoinEmptyResult)
{
    Context::getInstance();
    const std::string dbName = "JoinTestDb";
    const std::string tableAName = "TableA";
    const std::string tableBName = "TableB";
    const int32_t blockSize = 32; // length of a block

    std::vector<int32_t> idsA = {1};
    std::vector<int32_t> valuesA = {50};
    std::vector<int32_t> idsB = {1, 1};
    std::vector<int32_t> valuesB = {32, 33};


    std::shared_ptr<Database> joinDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
    Database::AddToInMemoryDatabaseList(joinDatabase);


    auto columnsA = std::unordered_map<std::string, DataType>();
    columnsA.insert(std::make_pair<std::string, DataType>("id", DataType::COLUMN_INT));
    columnsA.insert(std::make_pair<std::string, DataType>("value", DataType::COLUMN_INT));
    joinDatabase->CreateTable(columnsA, tableAName.c_str());

    auto columnsB = std::unordered_map<std::string, DataType>();
    columnsB.insert(std::make_pair<std::string, DataType>("id", DataType::COLUMN_INT));
    columnsB.insert(std::make_pair<std::string, DataType>("value", DataType::COLUMN_INT));
    joinDatabase->CreateTable(columnsB, tableBName.c_str());


    reinterpret_cast<ColumnBase<int32_t>*>(
        joinDatabase->GetTables().at(tableAName).GetColumns().at("id").get())
        ->InsertData(idsA);
    reinterpret_cast<ColumnBase<int32_t>*>(
        joinDatabase->GetTables().at(tableAName).GetColumns().at("value").get())
        ->InsertData(valuesA);

    reinterpret_cast<ColumnBase<int32_t>*>(
        joinDatabase->GetTables().at(tableBName).GetColumns().at("id").get())
        ->InsertData(idsB);
    reinterpret_cast<ColumnBase<int32_t>*>(
        joinDatabase->GetTables().at(tableBName).GetColumns().at("value").get())
        ->InsertData(valuesB);

    GpuSqlCustomParser parser(joinDatabase, "SELECT TableA.value, TableB.value FROM TableA JOIN "
                                            "TableB ON TableA.id = TableB.id;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloadA = result->payloads().at("TableA.value");
    auto& payloadB = result->payloads().at("TableB.value");

    ASSERT_EQ(2, payloadA.intpayload().intdata_size());
    ASSERT_EQ(2, payloadB.intpayload().intdata_size());

    // TODO assert values

    Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
}

TEST(DispatcherTestsRegression, AggregationInWhereWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "SELECT colInteger1, COUNT(colInteger2) FROM TableA WHERE "
                              "COUNT(colInteger2) > 10 GROUP "
                              "BY colInteger1;");
    ASSERT_THROW(parser.Parse(), AggregationWhereException);
}

TEST(DispatcherTestsRegression, CreateIndexWrongSemantic)
{
    Context::getInstance();

    GpuSqlCustomParser parser(DispatcherObjs::GetInstance().database,
                              "CREATE TABLE tblA (colA GEO_POINT, INDEX ind (colA));");
    ASSERT_THROW(parser.Parse(), IndexColumnDataTypeException);

    GpuSqlCustomParser parser2(DispatcherObjs::GetInstance().database,
                               "CREATE TABLE tblA (colA GEO_POLYGON, INDEX ind (colA));");
    ASSERT_THROW(parser2.Parse(), IndexColumnDataTypeException);
}
