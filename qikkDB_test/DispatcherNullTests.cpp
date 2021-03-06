#include "gtest/gtest.h"
#include "../qikkDB/DataType.h"
#include "../qikkDB/Database.h"
#include "../qikkDB/Table.h"
#include "../qikkDB/DatabaseGenerator.h"
#include "../qikkDB/GpuSqlParser/GpuSqlCustomParser.h"
#include "../qikkDB/ColumnBase.h"
#include "../qikkDB/PointFactory.h"
#include "../qikkDB/ComplexPolygonFactory.h"
#include "../qikkDB/GpuSqlParser/ParserExceptions.h"
#include "../qikkDB/messages/QueryResponseMessage.pb.h"
#include <sstream>


TEST(DispatcherNullTests, SelectNullWithWhere)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 1 << 5;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::vector<int> expectedResults;
    for (int i = 0; i < 16; i++)
    {
        if (i % 2 == i / 8)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
            parser.Parse();
        }
        else
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
            parser.Parse();
            expectedResults.push_back(1);
        }
    }
    GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 = 1;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
    ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
    for (int i = 0; i < payload.intpayload().intdata_size(); i++)
    {
        ASSERT_FLOAT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, IsNullWithPattern)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 1 << 5;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::vector<nullmask_t> expectedMask;
    nullmask_t bitMaskPart = 0;
    int nullCount = 0;
    for (int i = 0; i < 16; i++)
    {
        if (i % 2 == i / 8)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
            parser.Parse();
            bitMaskPart |= 1 << nullCount++;
        }
        else
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
            parser.Parse();
        }
    }
    expectedMask.push_back(bitMaskPart);
    GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 IS NULL;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& nullBitMask = result->nullbitmasks().at("TestTable.Col1").nullmask();
    // ASSERT_EQ(nullBitMask.size(), expectedMask.size());
    for (int i = 0; i < expectedMask.size(); i++)
    {
        ASSERT_EQ(expectedMask[i], nullBitMask[i]);
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}


TEST(DispatcherNullTests, IsNotNullWithPattern)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 1 << 5;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb"));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::vector<int> expectedResults;
    for (int i = 0; i < 16; i++)
    {
        if (i % 2 == i / 8)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
            parser.Parse();
        }
        else
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (1);");
            parser.Parse();
            expectedResults.push_back(1);
        }
    }
    GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable WHERE Col1 IS NOT NULL;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
    ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
    for (int i = 0; i < payload.intpayload().intdata_size(); i++)
    {
        ASSERT_FLOAT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}


TEST(DispatcherNullTests, OrderByNullTest)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbOrderByNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbOrderByNULL"));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    database->CreateTable(columns, "TestTable");

    std::vector<int32_t> expectedResults;
    std::vector<int8_t> expectedNullResults;

    for (int32_t i = 0; i < blockSize; i++)
    {
        if (i % 2)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1) VALUES (null);");
            parser.Parse();

            expectedResults.push_back(std::numeric_limits<int32_t>::lowest());
        }
        else
        {
            int32_t val = rand() % 1000;

            GpuSqlCustomParser parser(database,
                                      std::string("INSERT INTO TestTable (Col1) VALUES (") +
                                          std::to_string(val) + std::string(");"));
            parser.Parse();

            expectedResults.push_back(val);
        }

        if (i < blockSize / 2)
        {
            expectedNullResults.push_back(1);
        }
        else
        {
            expectedNullResults.push_back(0);
        }
    }

    GpuSqlCustomParser parser(database, "SELECT Col1 FROM TestTable ORDER BY Col1;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());
    auto& payload = result->payloads().at("TestTable.Col1");
    auto& nullBitMask = result->nullbitmasks().at("TestTable.Col1").nullmask();

    std::stable_sort(expectedResults.begin(), expectedResults.end());

    ASSERT_EQ(payload.intpayload().intdata_size(), expectedResults.size());
    for (int32_t i = 0; i < expectedResults.size(); i++)
    {
        nullmask_t nullBit = NullValues::GetConcreteBitFromBitmask(nullBitMask.begin(), i);
        if (!nullBit)
        {
            ASSERT_EQ(expectedResults[i], payload.intpayload().intdata()[i]);
        }
    }

    Database::RemoveFromInMemoryDatabaseList("TestDbOrderByNULL");
}

TEST(DispatcherNullTests, LimitOffsetNoClausesNoFullBlockNullTest)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbLimitOffsetNULL"));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    columns.emplace("Col2", COLUMN_STRING);
    columns.emplace("Col3", COLUMN_POINT);
    columns.emplace("Col4", COLUMN_POLYGON);
    database->CreateTable(columns, "TestTable");

    std::vector<int32_t> expectedResults1;
    std::vector<std::string> expectedResults2;
    std::vector<std::string> expectedResults3;
    std::vector<std::string> expectedResults4;

    for (int32_t i = 0; i < 17; i++)
    {
        if (i % 2)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1, Col2, Col3, Col4) "
                                                "VALUES (null,null, null, null);");
            parser.Parse();

            if (i > 6 && expectedResults1.size() < 9)
            {
                expectedResults1.push_back(0);
                expectedResults2.push_back("0");
                expectedResults3.push_back("0");
                expectedResults4.push_back("0");
            }
        }
        else
        {
            int32_t val = rand() % 1000;
            std::string valString = std::to_string(val);

            std::stringstream ssPoint;
            std::stringstream ssPolygon;

            ssPoint << "POINT(" << valString << " " << valString << ")";
            ssPolygon << "POLYGON((" << valString << " " << valString << ", " << valString << " "
                      << valString << "))";

            std::stringstream ssQuery;

            ssQuery << "INSERT INTO TestTable (Col1, Col2, Col3, Col4) VALUES (" << valString << ", "
                    << "\"" << valString << "\""
                    << ", " << ssPoint.str() << ", " << ssPolygon.str() << ");";

            GpuSqlCustomParser parser(database, ssQuery.str());
            parser.Parse();

            if (i > 6 && expectedResults1.size() < 9)
            {
                expectedResults1.push_back(val);
                expectedResults2.push_back(valString);
                expectedResults3.push_back(
                    PointFactory::WktFromPoint(PointFactory::FromWkt(ssPoint.str()), true));
                expectedResults4.push_back(
                    ComplexPolygonFactory::WktFromPolygon(ComplexPolygonFactory::FromWkt(ssPolygon.str()), true));
            }
        }
    }

    GpuSqlCustomParser parser(database,
                              "SELECT Col1, Col2, Col3, Col4 FROM TestTable LIMIT 9 OFFSET 7;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());

    auto& payload1 = result->payloads().at("TestTable.Col1");
    auto& nullBitMask1 = result->nullbitmasks().at("TestTable.Col1").nullmask();

    auto& payload2 = result->payloads().at("TestTable.Col2");
    auto& nullBitMask2 = result->nullbitmasks().at("TestTable.Col2").nullmask();

    auto& payload3 = result->payloads().at("TestTable.Col3");
    auto& nullBitMask3 = result->nullbitmasks().at("TestTable.Col3").nullmask();

    auto& payload4 = result->payloads().at("TestTable.Col4");
    auto& nullBitMask4 = result->nullbitmasks().at("TestTable.Col4").nullmask();


    ASSERT_EQ(payload1.intpayload().intdata_size(), expectedResults1.size());
    ASSERT_EQ(payload2.stringpayload().stringdata_size(), expectedResults2.size());
    ASSERT_EQ(payload3.stringpayload().stringdata_size(), expectedResults3.size());
    ASSERT_EQ(payload4.stringpayload().stringdata_size(), expectedResults4.size());

    for (int32_t i = 0; i < expectedResults1.size(); i++)
    {
        nullmask_t nullBit1 = NullValues::GetConcreteBitFromBitmask(nullBitMask1.begin(), i);
        if (!nullBit1)
        {
            ASSERT_EQ(expectedResults1[i], payload1.intpayload().intdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults1[i], 0);
        }

        nullmask_t nullBit2 = NullValues::GetConcreteBitFromBitmask(nullBitMask2.begin(), i);
        if (!nullBit2)
        {
            ASSERT_EQ(expectedResults2[i], payload2.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults2[i], "0");
        }

        nullmask_t nullBit3 = NullValues::GetConcreteBitFromBitmask(nullBitMask3.begin(), i);
        if (!nullBit3)
        {
            ASSERT_EQ(expectedResults3[i], payload3.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults3[i], "0");
        }

        nullmask_t nullBit4 = NullValues::GetConcreteBitFromBitmask(nullBitMask4.begin(), i);
        if (!nullBit4)
        {
            ASSERT_EQ(expectedResults4[i], payload4.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults4[i], "0");
        }
    }

    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");
}

TEST(DispatcherNullTests, LimitOffsetNoClausesCrossBlockNullTest)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbLimitOffsetNULL", blockSize));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columns;
    columns.emplace("Col1", COLUMN_INT);
    columns.emplace("Col2", COLUMN_STRING);
    columns.emplace("Col3", COLUMN_POINT);
    columns.emplace("Col4", COLUMN_POLYGON);
    database->CreateTable(columns, "TestTable");

    std::vector<int32_t> expectedResults1;
    std::vector<std::string> expectedResults2;
    std::vector<std::string> expectedResults3;
    std::vector<std::string> expectedResults4;

    int32_t limit = 10;
    int32_t offset = 28;

    for (int32_t i = 0; i < blockSize * 1.5; i++)
    {
        if (i % 2)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (Col1, Col2, Col3, Col4) "
                                                "VALUES (null,null, null, null);");
            parser.Parse();

            if (i > offset - 1 && expectedResults1.size() < limit)
            {
                expectedResults1.push_back(0);
                expectedResults2.push_back("0");
                expectedResults3.push_back("0");
                expectedResults4.push_back("0");
            }
        }
        else
        {
            int32_t val = rand() % 1000;
            std::string valString = std::to_string(val);

            std::stringstream ssPoint;
            std::stringstream ssPolygon;

            ssPoint << "POINT(" << valString << " " << valString << ")";
            ssPolygon << "POLYGON((" << valString << " " << valString << ", " << valString << " "
                      << valString << "))";

            std::stringstream ssQuery;

            ssQuery << "INSERT INTO TestTable (Col1, Col2, Col3, Col4) VALUES (" << valString << ", "
                    << "\"" << valString << "\""
                    << ", " << ssPoint.str() << ", " << ssPolygon.str() << ");";

            GpuSqlCustomParser parser(database, ssQuery.str());
            parser.Parse();

            if (i > offset - 1 && expectedResults1.size() < limit)
            {
                expectedResults1.push_back(val);
                expectedResults2.push_back(valString);
                expectedResults3.push_back(
                    PointFactory::WktFromPoint(PointFactory::FromWkt(ssPoint.str()), true));
                expectedResults4.push_back(
                    ComplexPolygonFactory::WktFromPolygon(ComplexPolygonFactory::FromWkt(ssPolygon.str()), true));
            }
        }
    }

    GpuSqlCustomParser parser(database, "SELECT Col1, Col2, Col3, Col4 FROM TestTable LIMIT " +
                                            std::to_string(limit) + " OFFSET " + std::to_string(offset) + ";");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());

    auto& payload1 = result->payloads().at("TestTable.Col1");
    auto& nullBitMask1 = result->nullbitmasks().at("TestTable.Col1").nullmask();

    auto& payload2 = result->payloads().at("TestTable.Col2");
    auto& nullBitMask2 = result->nullbitmasks().at("TestTable.Col2").nullmask();

    auto& payload3 = result->payloads().at("TestTable.Col3");
    auto& nullBitMask3 = result->nullbitmasks().at("TestTable.Col3").nullmask();

    auto& payload4 = result->payloads().at("TestTable.Col4");
    auto& nullBitMask4 = result->nullbitmasks().at("TestTable.Col4").nullmask();


    ASSERT_EQ(payload1.intpayload().intdata_size(), expectedResults1.size());
    ASSERT_EQ(payload2.stringpayload().stringdata_size(), expectedResults2.size());
    ASSERT_EQ(payload3.stringpayload().stringdata_size(), expectedResults3.size());
    ASSERT_EQ(payload4.stringpayload().stringdata_size(), expectedResults4.size());

    for (int32_t i = 0; i < expectedResults1.size(); i++)
    {
        int8_t nullBit1 = (nullBitMask1[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) & 1;
        if (!nullBit1)
        {
            ASSERT_EQ(expectedResults1[i], payload1.intpayload().intdata()[i]) << i;
        }
        else
        {
            ASSERT_EQ(expectedResults1[i], 0);
        }

        int8_t nullBit2 = (nullBitMask2[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) & 1;
        if (!nullBit2)
        {
            ASSERT_EQ(expectedResults2[i], payload2.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults2[i], "0");
        }

        int8_t nullBit3 = (nullBitMask3[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) & 1;
        if (!nullBit3)
        {
            ASSERT_EQ(expectedResults3[i], payload3.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults3[i], "0");
        }

        int8_t nullBit4 = (nullBitMask4[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) & 1;
        if (!nullBit4)
        {
            ASSERT_EQ(expectedResults4[i], payload4.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults4[i], "0");
        }
    }

    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");
}

TEST(DispatcherNullTests, JoinNullTestJoinOnNotNullTables)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbJoinNULL"));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columnsR;
    columnsR.emplace("ColA", COLUMN_INT);
    columnsR.emplace("ColJoinA", COLUMN_INT);

    std::unordered_map<std::string, DataType> columnsS;
    columnsS.emplace("ColB", COLUMN_INT);
    columnsS.emplace("ColJoinB", COLUMN_INT);

    database->CreateTable(columnsR, "TestTableR");
    database->CreateTable(columnsS, "TestTableS");

    for (int32_t i = 0, j = blockSize - 1; i < blockSize; i++, j--)
    {
        if (i % 2)
        {
            {
                GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA, "
                                                                "ColJoinA) VALUES (null,") +
                                                        std::to_string(i) + std::string(");"));
                parser.Parse();
            }
            {
                GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB, "
                                                                "ColJoinB) VALUES (") +
                                                        std::to_string(j) + std::string(",") +
                                                        std::to_string(j) + std::string(");"));
                parser.Parse();
            }
        }
        else
        {
            {
                GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableR (ColA, "
                                                                "ColJoinA) VALUES (") +
                                                        std::to_string(i) + std::string(",") +
                                                        std::to_string(i) + std::string(");"));
                parser.Parse();
            }
            {
                GpuSqlCustomParser parser(database, std::string("INSERT INTO TestTableS (ColB, "
                                                                "ColJoinB) VALUES (null,") +
                                                        std::to_string(j) + std::string(");"));
                parser.Parse();
            }
        }
    }

    GpuSqlCustomParser parser(database, "SELECT TestTableR.ColA, TestTableS.ColB FROM TestTableR "
                                        "JOIN TestTableS ON ColJoinA = ColJoinB;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());


    auto ColA = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTableR").GetColumns().at("ColA").get());
    auto ColB = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTableS").GetColumns().at("ColB").get());


    auto& payloadA = result->payloads().at("TestTableR.ColA");
    auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA").nullmask();

    auto& payloadB = result->payloads().at("TestTableS.ColB");
    auto& nullBitMaskB = result->nullbitmasks().at("TestTableS.ColB").nullmask();

    ASSERT_EQ(payloadA.intpayload().intdata_size(), payloadB.intpayload().intdata_size());
    for (int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
    {
        nullmask_t nullBitA = NullValues::GetConcreteBitFromBitmask(nullBitMaskA.begin(), i);
        nullmask_t nullBitB = NullValues::GetConcreteBitFromBitmask(nullBitMaskB.begin(), i);

        ASSERT_EQ(nullBitA, nullBitB);

        if (!nullBitA)
        {
            ASSERT_EQ(payloadA.intpayload().intdata()[i], payloadB.intpayload().intdata()[i]);
        }
    }


    Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL");
}

TEST(DispatcherNullTests, LimitOffsetClausesFullBlockNullTest)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbLimitOffsetNULL"));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columns;
    columns.emplace("ColA", COLUMN_INT);
    columns.emplace("Col1", COLUMN_INT);
    columns.emplace("Col2", COLUMN_STRING);
    columns.emplace("Col3", COLUMN_POINT);
    columns.emplace("Col4", COLUMN_POLYGON);
    database->CreateTable(columns, "TestTable");

    std::vector<int32_t> expectedResults1;
    std::vector<std::string> expectedResults2;
    std::vector<std::string> expectedResults3;
    std::vector<std::string> expectedResults4;

    for (int32_t i = 0; i < 17; i++)
    {
        if (i % 2)
        {
            GpuSqlCustomParser parser(database, "INSERT INTO TestTable (ColA, Col1, Col2, Col3, Col4) "
                                                "VALUES (1, null, null, null, null);");
            parser.Parse();

            if (i > 2 && expectedResults1.size() < 9)
            {
                expectedResults1.push_back(0);
                expectedResults2.push_back("0");
                expectedResults3.push_back("0");
                expectedResults4.push_back("0");
            }
        }
        else
        {
            int32_t val = 1;
            std::string valString = std::to_string(1);

            std::stringstream ssPoint;
            std::stringstream ssPolygon;

            ssPoint << "POINT(" << valString << " " << valString << ")";
            ssPolygon << "POLYGON((" << valString << " " << valString << ", " << valString << " "
                      << valString << "))";

            std::stringstream ssQuery;

            ssQuery << "INSERT INTO TestTable (ColA, Col1, Col2, Col3, Col4) VALUES (1, " << valString << ", "
                    << "\"" << valString << "\""
                    << ", " << ssPoint.str() << ", " << ssPolygon.str() << ");";

            GpuSqlCustomParser parser(database, ssQuery.str());
            parser.Parse();

            if (i > 2 && expectedResults1.size() < 9)
            {
                expectedResults1.push_back(val);
                expectedResults2.push_back(valString);
                expectedResults3.push_back(
                    PointFactory::WktFromPoint(PointFactory::FromWkt(ssPoint.str()), true));
                expectedResults4.push_back(
                    ComplexPolygonFactory::WktFromPolygon(ComplexPolygonFactory::FromWkt(ssPolygon.str()), true));
            }
        }
    }

    GpuSqlCustomParser parser(database,
                              "SELECT Col1, Col2, Col3, Col4 FROM TestTable WHERE ColA = 1 LIMIT 9 OFFSET 3;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto column = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTable").GetColumns().at("Col1").get());

    auto& payload1 = result->payloads().at("TestTable.Col1");
    auto& nullBitMask1 = result->nullbitmasks().at("TestTable.Col1").nullmask();

    auto& payload2 = result->payloads().at("TestTable.Col2");
    auto& nullBitMask2 = result->nullbitmasks().at("TestTable.Col2").nullmask();

    auto& payload3 = result->payloads().at("TestTable.Col3");
    auto& nullBitMask3 = result->nullbitmasks().at("TestTable.Col3").nullmask();

    auto& payload4 = result->payloads().at("TestTable.Col4");
    auto& nullBitMask4 = result->nullbitmasks().at("TestTable.Col4").nullmask();


    ASSERT_EQ(payload1.intpayload().intdata_size(), expectedResults1.size());
    ASSERT_EQ(payload2.stringpayload().stringdata_size(), expectedResults2.size());
    ASSERT_EQ(payload3.stringpayload().stringdata_size(), expectedResults3.size());
    ASSERT_EQ(payload4.stringpayload().stringdata_size(), expectedResults4.size());

    for (int32_t i = 0; i < expectedResults1.size(); i++)
    {
        nullmask_t nullBit1 = NullValues::GetConcreteBitFromBitmask(nullBitMask1.begin(), i);
        if (!nullBit1)
        {
            ASSERT_EQ(expectedResults1[i], payload1.intpayload().intdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults1[i], 0);
        }

        nullmask_t nullBit2 = NullValues::GetConcreteBitFromBitmask(nullBitMask2.begin(), i);
        if (!nullBit2)
        {
            ASSERT_EQ(expectedResults2[i], payload2.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults2[i], "0");
        }

        nullmask_t nullBit3 = NullValues::GetConcreteBitFromBitmask(nullBitMask3.begin(), i);
        if (!nullBit3)
        {
            ASSERT_EQ(expectedResults3[i], payload3.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults3[i], "0");
        }

        nullmask_t nullBit4 = NullValues::GetConcreteBitFromBitmask(nullBitMask4.begin(), i);
        if (!nullBit4)
        {
            ASSERT_EQ(expectedResults4[i], payload4.stringpayload().stringdata()[i]);
        }
        else
        {
            ASSERT_EQ(expectedResults4[i], "0");
        }
    }

    Database::RemoveFromInMemoryDatabaseList("TestDbLimitOffsetNULL");
}

/*
TEST(DispatcherNullTests, JoinIsNotNullTest)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL2");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database_(std::make_shared<Database>("TestDbJoinNULL2"));
    Database::AddToInMemoryDatabaseList(database_);

    std::unordered_map<std::string, DataType> columnsR;
    columnsR.emplace("ColA", COLUMN_INT);
    columnsR.emplace("ColJoinA", COLUMN_INT);

    std::unordered_map<std::string, DataType> columnsS;
    columnsS.emplace("ColB", COLUMN_INT);
    columnsS.emplace("ColJoinB", COLUMN_INT);

    database_->CreateTable(columnsR, "TestTableR");
    database_->CreateTable(columnsS, "TestTableS");

    for (int32_t i = 0, j = blockSize - 1; i < blockSize; i++, j--)
    {
        if (i % 2)
        {
            {
                GpuSqlCustomParser parser(database_, std::string("INSERT INTO TestTableR (ColA,
ColJoinA) VALUES (null,") + std::to_string(i) + std::string(");")); parser.parse();
            }
            {
                GpuSqlCustomParser parser(database_, std::string("INSERT INTO TestTableS (ColB,
ColJoinB) VALUES (") + std::to_string(j) + std::string(",") + std::to_string(j) +
std::string(");")); parser.parse();
            }
        }
        else
        {
            {
                GpuSqlCustomParser parser(database_, std::string("INSERT INTO TestTableR (ColA,
ColJoinA) VALUES (") + std::to_string(i) + std::string(",") + std::to_string(i) +
std::string(");")); parser.parse();
            }
            {
                GpuSqlCustomParser parser(database_, std::string("INSERT INTO TestTableS (ColB,
ColJoinB) VALUES (null,") + std::to_string(j) + std::string(");")); parser.parse();
            }
        }
    }

    GpuSqlCustomParser parser(database_, "SELECT TestTableR.ColA FROM TestTableR JOIN TestTableS ON
ColJoinA = ColJoinB WHERE TestTableR.ColA IS NOT NULL;"); auto resultPtr = parser.Parse(); auto
result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    auto ColA =
dynamic_cast<ColumnBase<int32_t>*>(database_->GetTables().at("TestTableR").GetColumns().at("ColA").get());
    auto ColJoinA =
dynamic_cast<ColumnBase<int32_t>*>(database_->GetTables().at("TestTableR").GetColumns().at("ColJoinA").get());
    auto ColJoinB =
dynamic_cast<ColumnBase<int32_t>*>(database_->GetTables().at("TestTableS").GetColumns().at("ColJoinB").get());

    std::vector<int32_t> expectedResults;

    for (int32_t leftBlockIdx = 0; leftBlockIdx < ColJoinA->GetBlockCount(); leftBlockIdx++)
    {
        auto leftRetBlock = ColA->GetBlocksList()[leftBlockIdx];
        auto leftBlock = ColJoinA->GetBlocksList()[leftBlockIdx];
        for (int32_t leftRowIdx = 0; leftRowIdx < leftBlock->GetSize(); leftRowIdx++)
        {
            for (int32_t rightBlockIdx = 0; rightBlockIdx < ColJoinB->GetBlockCount();
rightBlockIdx++)
            {
                auto rightBlock = ColJoinB->GetBlocksList()[rightBlockIdx];
                for (int32_t rightRowIdx = 0; rightRowIdx < rightBlock->GetSize(); rightRowIdx++)
                {
                    int8_t nullBit = (leftRetBlock->GetNullBitmask()[leftRowIdx / (sizeof(int8_t) *
8)] >> (leftRowIdx % (sizeof(int8_t) * 8))) & 1; if (leftBlock->GetData()[leftRowIdx] ==
rightBlock->GetData()[rightRowIdx] && nullBit == 0)
                    {
                        expectedResults.push_back(leftRetBlock->GetData()[leftRowIdx]);
                    }
                }
            }
        }
    }

    auto& payloadA = result->payloads().at("TestTableR.ColA");
    auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA");

    ASSERT_EQ(payloadA.intpayload().intdata().size(), expectedResults.size());

    for (int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
    {
        int8_t nullBitA = (nullBitMaskA[i / (sizeof(int8_t) * 8)] >> (i % (sizeof(int8_t) * 8))) &
1;

        ASSERT_EQ(payloadA.intpayload().intdata()[i], expectedResults[i]);
        ASSERT_EQ(nullBitA, 0);
    }


    Database::RemoveFromInMemoryDatabaseList("TestDbJoinNULL2");
}*/

TEST(DispatcherNullTests, JoinNullTestJoinOnNullTables)
{
    srand(42);
    Database::RemoveFromInMemoryDatabaseList("TestDbJoinOnNULL");

    int32_t blockSize = 1 << 5;

    std::shared_ptr<Database> database(std::make_shared<Database>("TestDbJoinOnNULL"));
    Database::AddToInMemoryDatabaseList(database);

    std::unordered_map<std::string, DataType> columnsR;
    columnsR.emplace("ColA", COLUMN_INT);

    std::unordered_map<std::string, DataType> columnsS;
    columnsS.emplace("ColB", COLUMN_INT);

    database->CreateTable(columnsR, "TestTableR");
    database->CreateTable(columnsS, "TestTableS");

    for (int32_t i = 0; i < blockSize; i++)
    {
        if (i % 2)
        {
            GpuSqlCustomParser parser(database,
                                      std::string("INSERT INTO TestTableR (ColA) VALUES (null);"));
            parser.Parse();
        }
        else
        {
            GpuSqlCustomParser parser(database,
                                      std::string("INSERT INTO TestTableR (ColA) VALUES (") +
                                          std::to_string(i) + std::string(");"));
            parser.Parse();
        }

        if (i < blockSize / 2)
        {
            GpuSqlCustomParser parser(database,
                                      std::string("INSERT INTO TestTableS (ColB) VALUES (null);"));
            parser.Parse();
        }
        else
        {
            GpuSqlCustomParser parser(database,
                                      std::string("INSERT INTO TestTableS (ColB) VALUES (") +
                                          std::to_string(i) + std::string(");"));
            parser.Parse();
        }
    }

    GpuSqlCustomParser parser(database, "SELECT TestTableR.ColA, TestTableS.ColB FROM TestTableR "
                                        "JOIN TestTableS ON ColA = ColB;");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

    auto ColA = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTableR").GetColumns().at("ColA").get());
    auto ColB = dynamic_cast<ColumnBase<int32_t>*>(
        database->GetTables().at("TestTableS").GetColumns().at("ColB").get());

    auto& payloadA = result->payloads().at("TestTableR.ColA");
    auto& nullBitMaskA = result->nullbitmasks().at("TestTableR.ColA").nullmask();

    auto& payloadB = result->payloads().at("TestTableS.ColB");
    auto& nullBitMaskB = result->nullbitmasks().at("TestTableS.ColB").nullmask();

    ASSERT_EQ(payloadA.intpayload().intdata_size(), payloadB.intpayload().intdata_size());
    for (int32_t i = 0; i < payloadA.intpayload().intdata_size(); i++)
    {
        nullmask_t nullBitA =
            (nullBitMaskA[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) &
            static_cast<nullmask_t>(1U);
        nullmask_t nullBitB =
            (nullBitMaskB[i / (sizeof(nullmask_t) * 8)] >> (i % (sizeof(nullmask_t) * 8))) &
            static_cast<nullmask_t>(1U);

        ASSERT_EQ(nullBitA, nullBitB);

        if (!nullBitA)
        {
            ASSERT_EQ(payloadA.intpayload().intdata()[i], payloadB.intpayload().intdata()[i]);
        }
    }


    Database::RemoveFromInMemoryDatabaseList("TestDbJoinOnNULL");
}

// == GROUP BY ==
TEST(DispatcherNullTests, GroupByNullKeySum)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<int32_t, int32_t> expectedResults;
    int32_t expectedValueAtNull = 0;
    for (int i = 0; i < 32; i++)
    {
        bool nullKey = (i % 4 == 2);
        int32_t intKey = i % 4;
        int32_t intVal = 2;
        std::string key = (nullKey ? "NULL" : std::to_string(intKey));
        std::string val = std::to_string(intVal);
        if (nullKey)
        {
            expectedValueAtNull += intVal;
        }
        else
        {
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, intVal});
            }
            else
            {
                expectedResults[intKey] += intVal;
            }
        }
        std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");")
                  << std::endl;
        GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" +
                                                key + ", " + val + ");");
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  NULL    | 16
    //  0       | 16
    //  1       | 16
    //  3       | 16
    int numberOfNullKeys = 0;
    for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        ASSERT_FALSE(valIsNull);
        if (keyIsNull)
        {
            ASSERT_EQ(expectedValueAtNull, valuesResult.intpayload().intdata()[i]);
            numberOfNullKeys++;
        }
        else
        {
            ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) ==
                         expectedResults.end());
            ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]),
                      valuesResult.intpayload().intdata()[i]);
        }
    }
    ASSERT_EQ(1, numberOfNullKeys);
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullKeyAvg)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<int32_t, int32_t> expectedResults;
    int32_t expectedValueAtNull = 0;
    for (int i = 0; i < 32; i++)
    {
        bool nullKey = (i % 4 == 2);
        int32_t intKey = i % 4;
        int32_t intVal = 2;
        std::string key = (nullKey ? "NULL" : std::to_string(intKey));
        std::string val = std::to_string(intVal);
        if (nullKey)
        {
            expectedValueAtNull = intVal;
        }
        else
        {
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, intVal});
            }
        }
        std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");")
                  << std::endl;
        GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" +
                                                key + ", " + val + ");");
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, AVG(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("AVG(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("AVG(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("AVG(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  NULL    | 2
    //  0       | 2
    //  1       | 2
    //  3       | 2
    int numberOfNullKeys = 0;
    for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        std::cout
            << "key: " << (keyIsNull ? "NULL" : std::to_string(keysResult.intpayload().intdata()[i]))
            << ", value:" << (valIsNull ? "NULL" : std::to_string(valuesResult.intpayload().intdata()[i]))
            << std::endl;
        ASSERT_FALSE(valIsNull)
            << "at key " << (keyIsNull ? "NULL" : std::to_string(keysResult.intpayload().intdata()[i]));
        if (keyIsNull)
        {
            ASSERT_EQ(expectedValueAtNull, valuesResult.intpayload().intdata()[i]);
            numberOfNullKeys++;
        }
        else
        {
            ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) ==
                         expectedResults.end());
            ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]),
                      valuesResult.intpayload().intdata()[i]);
        }
    }
    ASSERT_EQ(1, numberOfNullKeys);
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueSum)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<int32_t, int32_t> expectedResults;
    std::unordered_map<int32_t, bool> expectedValueNullMask;
    for (int i = 0; i < 32; i++)
    {
        int32_t intKey = i % 4;
        int32_t intVal = 2;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = std::to_string(intKey);
        std::string val = (nullValue ? "NULL" : std::to_string(intVal));
        if (nullValue)
        {
            if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({intKey, true});
            }
        }
        else
        {
            // "turn of" null
            if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({intKey, false});
            }
            else
            {
                expectedValueNullMask[intKey] = false;
            }
            // aggregate value
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, intVal});
            }
            else
            {
                expectedResults[intKey] += intVal;
            }
        }

        std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");")
                  << std::endl;
        GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" +
                                                key + ", " + val + ");");
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  0       | 16
    //  1       | 16
    //  2       | 8
    //  3       | NULL
    for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        const int32_t key = keysResult.intpayload().intdata()[i];
        // Check nulls
        ASSERT_FALSE(keyIsNull) << " at result row " << i;
        ASSERT_FALSE(expectedValueNullMask.find(keysResult.intpayload().intdata()[i]) ==
                     expectedValueNullMask.end())
            << " bad key at result row " << i;
        ASSERT_EQ(expectedValueNullMask.at(keysResult.intpayload().intdata()[i]), valIsNull)
            << " at result row " << i;
        if (!valIsNull)
        {
            // Check value
            ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) ==
                         expectedResults.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]),
                      valuesResult.intpayload().intdata()[i])
                << " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueAvg)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<int32_t, int32_t> expectedResults;
    std::unordered_map<int32_t, bool> expectedValueNullMask;
    const int INT_VAL_CONST = 2;
    for (int i = 0; i < 32; i++)
    {
        int32_t intKey = i % 4;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = std::to_string(intKey);
        std::string val = (nullValue ? "NULL" : std::to_string(INT_VAL_CONST));
        if (nullValue)
        {
            if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({intKey, true});
            }
        }
        else
        {
            // "turn of" null
            if (expectedValueNullMask.find(intKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({intKey, false});
            }
            else
            {
                expectedValueNullMask[intKey] = false;
            }
            // set value
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, INT_VAL_CONST});
            }
        }

        std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");")
                  << std::endl;
        GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" +
                                                key + ", " + val + ");");
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, AVG(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("AVG(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("AVG(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("AVG(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  0       | 2
    //  1       | 2
    //  2       | 2
    //  3       | NULL
    for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        const int32_t key = keysResult.intpayload().intdata()[i];
        // Check nulls
        ASSERT_FALSE(keyIsNull) << " at result row " << i;
        ASSERT_FALSE(expectedValueNullMask.find(keysResult.intpayload().intdata()[i]) ==
                     expectedValueNullMask.end())
            << " bad key at result row " << i;
        ASSERT_EQ(expectedValueNullMask.at(keysResult.intpayload().intdata()[i]), valIsNull)
            << " at result row " << i;
        if (!valIsNull)
        {
            // Check keys and values
            ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) ==
                         expectedResults.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]),
                      valuesResult.intpayload().intdata()[i])
                << " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByNullValueCount)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<int32_t, int64_t> expectedResults;
    for (int i = 0; i < 32; i++)
    {
        int32_t intKey = i % 4;
        int32_t intVal = 2;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = std::to_string(intKey);
        std::string val = (nullValue ? "NULL" : std::to_string(intVal));
        if (nullValue)
        {
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, 0});
            }
        }
        else
        {
            if (expectedResults.find(intKey) == expectedResults.end())
            {
                expectedResults.insert({intKey, 1});
            }
            else
            {
                expectedResults[intKey]++;
            }
        }

        std::cout << ("INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");")
                  << std::endl;
        GpuSqlCustomParser parser(database, "INSERT INTO TestTable (colKeys, colVals) VALUES (" +
                                                key + ", " + val + ");");
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, COUNT(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("COUNT(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("COUNT(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("COUNT(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  0       | 8
    //  1       | 8
    //  2       | 4
    //  3       | 0
    for (int i = 0; i < keysResult.intpayload().intdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        const int32_t key = keysResult.intpayload().intdata()[i];
        // Check nulls (there should be no null value)
        ASSERT_FALSE(keyIsNull) << " at result row " << i;
        ASSERT_FALSE(valIsNull) << " at result row " << i;
        // Check value
        ASSERT_FALSE(expectedResults.find(keysResult.intpayload().intdata()[i]) == expectedResults.end())
            << " bad key at result row " << i;
        ASSERT_EQ(expectedResults.at(keysResult.intpayload().intdata()[i]),
                  valuesResult.int64payload().int64data()[i])
            << " with key " << keysResult.intpayload().intdata()[i] << " at result row " << i;
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}


//== GROUP BY String ==
TEST(DispatcherNullTests, GroupByStringNullKeySum)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_STRING);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<std::string, int32_t> expectedResults;
    int32_t expectedValueAtNull = 0;
    for (int i = 0; i < 32; i++)
    {
        bool nullKey = (i % 4 == 2);
        std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : "XYZ"));
        int32_t intVal = 2;
        std::string key = (nullKey ? "NULL" : ("\"" + strKey + "\""));
        std::string val = std::to_string(intVal);
        if (nullKey)
        {
            expectedValueAtNull += intVal;
        }
        else
        {
            if (expectedResults.find(strKey) == expectedResults.end())
            {
                expectedResults.insert({strKey, intVal});
            }
            else
            {
                expectedResults[strKey] += intVal;
            }
        }
        std::string insertQuery = "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  NULL    | 16
    //  Apple   | 16
    //  Nut     | 16
    //  XYZ     | 16
    ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
    ASSERT_EQ(4, valuesResult.intpayload().intdata_size());
    for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        std::cout << i << ": " << (keyIsNull ? "-NULL-" : keysResult.stringpayload().stringdata()[i])
                  << " | " << valuesResult.intpayload().intdata()[i] << std::endl;
        ASSERT_FALSE(valIsNull);
        if (keyIsNull)
        {
            ASSERT_EQ(expectedValueAtNull, valuesResult.intpayload().intdata()[i]);
        }
        else
        {
            ASSERT_FALSE(expectedResults.find(keysResult.stringpayload().stringdata()[i]) ==
                         expectedResults.end())
                << keysResult.stringpayload().stringdata()[i];
            ASSERT_EQ(expectedResults.at(keysResult.stringpayload().stringdata()[i]),
                      valuesResult.intpayload().intdata()[i])
                << keysResult.stringpayload().stringdata()[i];
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByStringNullValueSum)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_STRING);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<std::string, int32_t> expectedResults;
    std::unordered_map<std::string, bool> expectedValueNullMask;
    for (int i = 0; i < 24; i++)
    {
        std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : (i % 4 == 2 ? "Straw" : "car0")));
        int32_t intVal = 2;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = "\"" + strKey + "\"";
        std::string val = (nullValue ? "NULL" : std::to_string(intVal));
        if (nullValue)
        {
            if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({strKey, true});
            }
        }
        else
        {
            // "turn of" null
            if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({strKey, false});
            }
            else
            {
                expectedValueNullMask[strKey] = false;
            }
            // aggregate value
            if (expectedResults.find(strKey) == expectedResults.end())
            {
                expectedResults.insert({strKey, intVal});
            }
            else
            {
                expectedResults[strKey] += intVal;
            }
        }
        std::string insertQuery = "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, SUM(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("SUM(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("SUM(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("SUM(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  Apple   | 12
    //  Nut     | 12
    //  Straw   | 8
    //  car0    | NULL
    ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
    ASSERT_EQ(4, valuesResult.intpayload().intdata_size());
    for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        std::string key = keysResult.stringpayload().stringdata()[i];
        std::cout << i << ": " << (keyIsNull ? "-NULL-" : key) << " | "
                  << (valIsNull ? "-NULL-" : std::to_string(valuesResult.intpayload().intdata()[i]))
                  << std::endl;
        // Check nulls
        ASSERT_FALSE(keyIsNull) << " at result row " << i;
        ASSERT_FALSE(expectedValueNullMask.find(key) == expectedValueNullMask.end())
            << " bad key at result row " << i;
        ASSERT_EQ(expectedValueNullMask.at(key), valIsNull) << " at result row " << i;
        if (!valIsNull)
        {
            // Check value
            ASSERT_FALSE(expectedResults.find(key) == expectedResults.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResults.at(key), valuesResult.intpayload().intdata()[i])
                << " with key " << key << " at result row " << i;
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByStringNullValueAvg)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_STRING);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<std::string, int32_t> expectedResults;
    std::unordered_map<std::string, bool> expectedValueNullMask;
    for (int i = 0; i < 24; i++)
    {
        std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : (i % 4 == 2 ? "Straw" : "car0")));
        int32_t intVal = 2;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = "\"" + strKey + "\"";
        std::string val = (nullValue ? "NULL" : std::to_string(intVal));
        if (nullValue)
        {
            if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({strKey, true});
            }
        }
        else
        {
            // "turn of" null
            if (expectedValueNullMask.find(strKey) == expectedValueNullMask.end())
            {
                expectedValueNullMask.insert({strKey, false});
            }
            else
            {
                expectedValueNullMask[strKey] = false;
            }
            // set value (this tests' case is average of the same numbers)
            if (expectedResults.find(strKey) == expectedResults.end())
            {
                expectedResults.insert({strKey, intVal});
            }
        }
        std::string insertQuery = "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, AVG(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("AVG(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("AVG(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("AVG(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  Apple   | 2
    //  Nut     | 2
    //  Straw   | 2
    //  car0    | NULL
    ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
    ASSERT_EQ(4, valuesResult.intpayload().intdata_size());
    for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        std::string key = keysResult.stringpayload().stringdata()[i];
        std::cout << i << ": " << (keyIsNull ? "-NULL-" : key) << " | "
                  << (valIsNull ? "-NULL-" : std::to_string(valuesResult.intpayload().intdata()[i]))
                  << std::endl;
        // Check nulls
        ASSERT_FALSE(keyIsNull) << " at result row " << i;
        ASSERT_FALSE(expectedValueNullMask.find(key) == expectedValueNullMask.end())
            << " bad key at result row " << i;
        ASSERT_EQ(expectedValueNullMask.at(key), valIsNull) << " at result row " << i;
        if (!valIsNull)
        {
            // Check value
            ASSERT_FALSE(expectedResults.find(key) == expectedResults.end())
                << " bad key at result row " << i;
            ASSERT_EQ(expectedResults.at(key), valuesResult.intpayload().intdata()[i])
                << " with key " << key << " at result row " << i;
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}

TEST(DispatcherNullTests, GroupByStringNullValueCount)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 8;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeys", COLUMN_STRING);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    std::unordered_map<std::string, int64_t> expectedResults;
    for (int i = 0; i < 24; i++)
    {
        std::string strKey = (i % 4 == 0 ? "Apple" : (i % 4 == 1 ? "Nut" : (i % 4 == 2 ? "Straw" : "car0")));
        int32_t intVal = 2;
        bool nullValue = (((i % 4) == 2 && (i < 8)) || ((i % 4) == 3));
        std::string key = "\"" + strKey + "\"";
        std::string val = (nullValue ? "NULL" : std::to_string(intVal));
        if (nullValue)
        {
            expectedResults.insert({strKey, 0});
        }
        else
        {
            // aggregate count
            if (expectedResults.find(strKey) == expectedResults.end())
            {
                expectedResults.insert({strKey, 1});
            }
            else
            {
                expectedResults[strKey] += 1;
            }
        }
        std::string insertQuery = "INSERT INTO TestTable (colKeys, colVals) VALUES (" + key + ", " + val + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }

    GpuSqlCustomParser parser(database,
                              "SELECT colKeys, COUNT(colVals) FROM TestTable GROUP BY colKeys;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeys"));
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("COUNT(colVals)"));
    auto& keysNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeys").nullmask();
    auto& valuesNullMaskResult = responseMessage->nullbitmasks().at("COUNT(colVals)").nullmask();
    auto& keysResult = responseMessage->payloads().at("TestTable.colKeys");
    auto& valuesResult = responseMessage->payloads().at("COUNT(colVals)");

    // Result should look like:
    //  colKeys | colVals
    //  Apple   | 6
    //  Nut     | 6
    //  Straw   | 4
    //  car0    | 0
    ASSERT_EQ(4, keysResult.stringpayload().stringdata_size());
    ASSERT_EQ(4, valuesResult.int64payload().int64data_size());
    for (int i = 0; i < keysResult.stringpayload().stringdata_size(); i++)
    {
        const char keyChar = keysNullMaskResult[i / 8];
        const bool keyIsNull = ((keyChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        std::string key = keysResult.stringpayload().stringdata()[i];
        std::cout << i << ": " << (keyIsNull ? "-NULL-" : key) << " | "
                  << (valIsNull ? "-NULL-" : std::to_string(valuesResult.int64payload().int64data()[i]))
                  << std::endl;
        // Check nulls
        ASSERT_FALSE(keyIsNull) << " at key " << key;
        ASSERT_FALSE(valIsNull) << " at key " << key;
        if (!valIsNull)
        {
            // Check value
            ASSERT_FALSE(expectedResults.find(key) == expectedResults.end()) << " bad key: " << key;
            ASSERT_EQ(expectedResults.at(key), valuesResult.int64payload().int64data()[i])
                << " with key " << key << " at result row " << i;
        }
    }
    Database::RemoveFromInMemoryDatabaseList("TestDb");
}


//== GROUP BY Multi-key ==
void TestGBMK(std::string aggregationOperation,
              std::vector<int32_t> keysA,
              std::vector<bool> keysANullMask,
              std::vector<int32_t> keysB,
              std::vector<bool> keysBNullMask,
              std::vector<int32_t> values,
              std::vector<bool> valuesNullMask,
              std::vector<int32_t> keysAExpectedResult,
              std::vector<bool> keysANullMaskExpectedResult,
              std::vector<int32_t> keysBExpectedResult,
              std::vector<bool> keysBNullMaskExpectedResult,
              std::vector<int32_t> valuesExpectedResult,
              std::vector<bool> valuesNullMaskExpectedResult)
{
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 4;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeysA", COLUMN_INT);
    columns.emplace("colKeysB", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    for (int i = 0; i < keysA.size(); i++)
    {
        std::string ka = keysANullMask[i] ? "NULL" : std::to_string(keysA[i]);
        std::string kb = keysBNullMask[i] ? "NULL" : std::to_string(keysB[i]);
        std::string vl = valuesNullMask[i] ? "NULL" : std::to_string(values[i]);
        std::string insertQuery = "INSERT INTO TestTable (colKeysA, colKeysB, colVals) VALUES (" +
                                  ka + ", " + kb + ", " + vl + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }
    GpuSqlCustomParser parser(database, "SELECT colKeysA, colKeysB, " + aggregationOperation +
                                            "(colVals) FROM TestTable "
                                            "GROUP BY colKeysA, colKeysB;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeysA"))
        << "colKeysA null mask does not exist";
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeysB"))
        << "colKeysB null mask does not exist";
    ASSERT_TRUE(responseMessage->nullbitmasks().contains(aggregationOperation + "(colVals)"))
        << "colVals null mask does not exist";
    auto& keysANullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeysA").nullmask();
    auto& keysBNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeysB").nullmask();
    auto& valuesNullMaskResult =
        responseMessage->nullbitmasks().at(aggregationOperation + "(colVals)").nullmask();
    auto& keysAResult = responseMessage->payloads().at("TestTable.colKeysA");
    auto& keysBResult = responseMessage->payloads().at("TestTable.colKeysB");
    auto& valuesResult = responseMessage->payloads().at(aggregationOperation + "(colVals)");

    ASSERT_EQ(keysAExpectedResult.size(), keysAResult.intpayload().intdata_size());
    ASSERT_EQ(keysBExpectedResult.size(), keysBResult.intpayload().intdata_size());
    ASSERT_EQ(valuesExpectedResult.size(), valuesResult.intpayload().intdata_size());

    for (int i = 0; i < keysAResult.intpayload().intdata_size(); i++)
    {
        const char keyAChar = keysANullMaskResult[i / 8];
        const bool keyAIsNull = ((keyAChar >> (i % 8)) & 1);
        const char keyBChar = keysBNullMaskResult[i / 8];
        const bool keyBIsNull = ((keyBChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        int foundIndex = -1; // index in expected result vectors
        for (int j = 0; j < keysAExpectedResult.size(); j++)
        {
            if (keysANullMaskExpectedResult[j] == keyAIsNull && keysBNullMaskExpectedResult[j] == keyBIsNull &&
                (keyAIsNull || keysAExpectedResult[j] == keysAResult.intpayload().intdata()[i]) &&
                (keyBIsNull || keysBExpectedResult[j] == keysBResult.intpayload().intdata()[i]))
            {
                foundIndex = j;
                break;
            }
        }
        ASSERT_NE(-1, foundIndex)
            << "key " << (keyAIsNull ? "NULL" : std::to_string(keysAResult.intpayload().intdata()[i]))
            << " " << (keyBIsNull ? "NULL" : std::to_string(keysBResult.intpayload().intdata()[i]))
            << " not found";
        std::cout
            << foundIndex
            << ", key: " << (keyAIsNull ? "NULL" : std::to_string(keysAResult.intpayload().intdata()[i]))
            << " " << (keyBIsNull ? "NULL" : std::to_string(keysBResult.intpayload().intdata()[i]))
            << ", value:" << (valIsNull ? "NULL" : std::to_string(valuesResult.intpayload().intdata()[i]))
            << std::endl;
        ASSERT_EQ(valuesNullMaskExpectedResult[foundIndex], valIsNull);
        if (!valIsNull)
        {
            ASSERT_EQ(valuesExpectedResult[foundIndex], valuesResult.intpayload().intdata()[i]);
        }
    }
}

void TestGBMKCount(std::vector<int32_t> keysA,
                   std::vector<bool> keysANullMask,
                   std::vector<int32_t> keysB,
                   std::vector<bool> keysBNullMask,
                   std::vector<int32_t> values,
                   std::vector<bool> valuesNullMask,
                   std::vector<int32_t> keysAExpectedResult,
                   std::vector<bool> keysANullMaskExpectedResult,
                   std::vector<int32_t> keysBExpectedResult,
                   std::vector<bool> keysBNullMaskExpectedResult,
                   std::vector<int64_t> valuesExpectedResult)
{
    const std::string aggregationOperation = "COUNT";
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 4;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colKeysA", COLUMN_INT);
    columns.emplace("colKeysB", COLUMN_INT);
    columns.emplace("colVals", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    for (int i = 0; i < keysA.size(); i++)
    {
        std::string ka = keysANullMask[i] ? "NULL" : std::to_string(keysA[i]);
        std::string kb = keysBNullMask[i] ? "NULL" : std::to_string(keysB[i]);
        std::string vl = valuesNullMask[i] ? "NULL" : std::to_string(values[i]);
        std::string insertQuery = "INSERT INTO TestTable (colKeysA, colKeysB, colVals) VALUES (" +
                                  ka + ", " + kb + ", " + vl + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }
    GpuSqlCustomParser parser(database, "SELECT colKeysA, colKeysB, " + aggregationOperation +
                                            "(colVals) FROM TestTable "
                                            "GROUP BY colKeysA, colKeysB;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeysA"))
        << "colKeysA null mask does not exist";
    ASSERT_TRUE(responseMessage->nullbitmasks().contains("TestTable.colKeysB"))
        << "colKeysB null mask does not exist";
    ASSERT_TRUE(responseMessage->nullbitmasks().contains(aggregationOperation + "(colVals)"))
        << "colVals null mask does not exist";
    auto& keysANullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeysA").nullmask();
    auto& keysBNullMaskResult = responseMessage->nullbitmasks().at("TestTable.colKeysB").nullmask();
    auto& valuesNullMaskResult =
        responseMessage->nullbitmasks().at(aggregationOperation + "(colVals)").nullmask();
    auto& keysAResult = responseMessage->payloads().at("TestTable.colKeysA");
    auto& keysBResult = responseMessage->payloads().at("TestTable.colKeysB");
    auto& valuesResult = responseMessage->payloads().at(aggregationOperation + "(colVals)");

    ASSERT_EQ(keysAExpectedResult.size(), keysAResult.intpayload().intdata_size());
    ASSERT_EQ(keysBExpectedResult.size(), keysBResult.intpayload().intdata_size());
    ASSERT_EQ(valuesExpectedResult.size(), valuesResult.int64payload().int64data_size());

    for (int i = 0; i < keysAResult.intpayload().intdata_size(); i++)
    {
        const char keyAChar = keysANullMaskResult[i / 8];
        const bool keyAIsNull = ((keyAChar >> (i % 8)) & 1);
        const char keyBChar = keysBNullMaskResult[i / 8];
        const bool keyBIsNull = ((keyBChar >> (i % 8)) & 1);
        const char valChar = valuesNullMaskResult[i / 8];
        const bool valIsNull = ((valChar >> (i % 8)) & 1);
        int foundIndex = -1; // index in expected result vectors
        for (int j = 0; j < keysAExpectedResult.size(); j++)
        {
            if (keysANullMaskExpectedResult[j] == keyAIsNull && keysBNullMaskExpectedResult[j] == keyBIsNull &&
                (keyAIsNull || keysAExpectedResult[j] == keysAResult.intpayload().intdata()[i]) &&
                (keyBIsNull || keysBExpectedResult[j] == keysBResult.intpayload().intdata()[i]))
            {
                foundIndex = j;
                break;
            }
        }
        ASSERT_NE(-1, foundIndex)
            << "key " << (keyAIsNull ? "NULL" : std::to_string(keysAResult.intpayload().intdata()[i]))
            << " " << (keyBIsNull ? "NULL" : std::to_string(keysBResult.intpayload().intdata()[i]))
            << " not found";
        std::cout << foundIndex << ", key: "
                  << (keyAIsNull ? "NULL" : std::to_string(keysAResult.intpayload().intdata()[i])) << " "
                  << (keyBIsNull ? "NULL" : std::to_string(keysBResult.intpayload().intdata()[i])) << ", value:"
                  << (valIsNull ? "NULL" : std::to_string(valuesResult.int64payload().int64data()[i]))
                  << std::endl;
        ASSERT_FALSE(valIsNull);
        ASSERT_EQ(valuesExpectedResult[foundIndex], valuesResult.int64payload().int64data()[i]);
    }
}

void TestOrEqualNullMaskMerging(std::vector<int32_t> colA,
                                std::vector<bool> colANullMask,
                                std::vector<int32_t> colB,
                                std::vector<bool> colBNullMask,
                                std::vector<int32_t> colC,
                                std::vector<int32_t> colCExpectedResult)
{
    const std::string aggregationOperation = "COUNT";
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 4;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colA", COLUMN_INT);
    columns.emplace("colB", COLUMN_INT);
    columns.emplace("colC", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    for (int i = 0; i < colA.size(); i++)
    {
        std::string ka = colANullMask[i] ? "NULL" : std::to_string(colA[i]);
        std::string kb = colBNullMask[i] ? "NULL" : std::to_string(colB[i]);
        std::string vl = std::to_string(colC[i]);
        std::string insertQuery =
            "INSERT INTO TestTable (colA, colB, colC) VALUES (" + ka + ", " + kb + ", " + vl + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }
    GpuSqlCustomParser parser(database, "SELECT colC FROM TestTable WHERE colA = 0 OR colB = 0;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& colCResult = responseMessage->payloads().at("TestTable.colC");

    ASSERT_EQ(colCExpectedResult.size(), colCResult.intpayload().intdata_size());

    for (int i = 0; i < colCResult.intpayload().intdata_size(); i++)
    {
        ASSERT_EQ(colCExpectedResult[i], colCResult.intpayload().intdata()[i]);
    }
}

void TestOrIsNullNullMaskMerging(std::vector<int32_t> colA,
                                 std::vector<bool> colANullMask,
                                 std::vector<int32_t> colB,
                                 std::vector<bool> colBNullMask,
                                 std::vector<int32_t> colC,
                                 std::vector<int32_t> colCExpectedResult)
{
    const std::string aggregationOperation = "COUNT";
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 5;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colA", COLUMN_INT);
    columns.emplace("colB", COLUMN_INT);
    columns.emplace("colC", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    for (int i = 0; i < colA.size(); i++)
    {
        std::string ka = colANullMask[i] ? "NULL" : std::to_string(colA[i]);
        std::string kb = colBNullMask[i] ? "NULL" : std::to_string(colB[i]);
        std::string vl = std::to_string(colC[i]);
        std::string insertQuery =
            "INSERT INTO TestTable (colA, colB, colC) VALUES (" + ka + ", " + kb + ", " + vl + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }
    GpuSqlCustomParser parser(database,
                              "SELECT colC FROM TestTable WHERE colA IS NULL OR colB IS NULL;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& colCResult = responseMessage->payloads().at("TestTable.colC");

    ASSERT_EQ(colCExpectedResult.size(), colCResult.intpayload().intdata_size());

    for (int i = 0; i < colCResult.intpayload().intdata_size(); i++)
    {
        ASSERT_EQ(colCExpectedResult[i], colCResult.intpayload().intdata()[i]);
    }
}

void TestOrNotNullMaskMerging(std::vector<int32_t> colA,
                              std::vector<bool> colANullMask,
                              std::vector<int32_t> colB,
                              std::vector<bool> colBNullMask,
                              std::vector<int32_t> colC,
                              std::vector<int32_t> colCExpectedResult)
{
    const std::string aggregationOperation = "COUNT";
    Database::RemoveFromInMemoryDatabaseList("TestDb");
    const int blockSize = 5;
    std::shared_ptr<Database> database(std::make_shared<Database>("TestDb", blockSize));
    Database::AddToInMemoryDatabaseList(database);
    std::unordered_map<std::string, DataType> columns;
    columns.emplace("colA", COLUMN_INT);
    columns.emplace("colB", COLUMN_INT);
    columns.emplace("colC", COLUMN_INT);
    database->CreateTable(columns, "TestTable");
    for (int i = 0; i < colA.size(); i++)
    {
        std::string ka = colANullMask[i] ? "NULL" : std::to_string(colA[i]);
        std::string kb = colBNullMask[i] ? "NULL" : std::to_string(colB[i]);
        std::string vl = std::to_string(colC[i]);
        std::string insertQuery =
            "INSERT INTO TestTable (colA, colB, colC) VALUES (" + ka + ", " + kb + ", " + vl + ");";
        std::cout << insertQuery << std::endl;
        GpuSqlCustomParser parser(database, insertQuery);
        parser.Parse();
    }
    GpuSqlCustomParser parser(database, "SELECT colC FROM TestTable WHERE !colA OR !colB;");
    auto resultPtr = parser.Parse();
    auto responseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& colCResult = responseMessage->payloads().at("TestTable.colC");

    ASSERT_EQ(colCExpectedResult.size(), colCResult.intpayload().intdata_size());

    for (int i = 0; i < colCResult.intpayload().intdata_size(); i++)
    {
        ASSERT_EQ(colCExpectedResult[i], colCResult.intpayload().intdata()[i]);
    }
}

// NULL keys tests
TEST(DispatcherNullTests, GroupByMultiKeyNullKeySum)
{
    TestGBMK("SUM", {-1, 0, -1, 0, 5, 1, -1, -1}, {true, false, true, false, false, false, true, true},
             {-1, -1, 0, 0, 3, -1, 3, -1}, {true, true, false, false, false, true, false, true},
             {10, 10, 10, 10, 10, 10, 10, 10}, {false, false, false, false, false, false, false, false},
             {-1, -1, -1, 0, 1, 0, 5}, {true, true, true, false, false, false, false},
             {-1, 0, 3, -1, -1, 0, 3}, {true, false, false, true, true, false, false},
             {20, 10, 10, 10, 10, 10, 10}, {false, false, false, false, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullKeyMin)
{
    TestGBMK("MIN", {-1, 0, -1, 0, 5, 1, -1, -1}, {true, false, true, false, false, false, true, true},
             {-1, -1, 0, 0, 3, -1, 3, -1}, {true, true, false, false, false, true, false, true},
             {-5, 80, 10, 10, 10, 10, 10, 30}, {false, false, false, false, false, false, false, false},
             {-1, -1, -1, 0, 1, 0, 5}, {true, true, true, false, false, false, false},
             {-1, 0, 3, -1, -1, 0, 3}, {true, false, false, true, true, false, false},
             {-5, 10, 10, 80, 10, 10, 10}, {false, false, false, false, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullKeyMax)
{
    TestGBMK("MAX", {-1, 0, -1, 0, 5, 1, -1, -1}, {true, false, true, false, false, false, true, true},
             {-1, -1, 0, 0, 3, -1, 3, -1}, {true, true, false, false, false, true, false, true},
             {-5, 80, 1, 2, 3, 4, 5, 30}, {false, false, false, false, false, false, false, false},
             {-1, -1, -1, 0, 1, 0, 5}, {true, true, true, false, false, false, false},
             {-1, 0, 3, -1, -1, 0, 3}, {true, false, false, true, true, false, false},
             {30, 1, 5, 80, 4, 2, 3}, {false, false, false, false, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullKeyAvg)
{
    TestGBMK("AVG", {-1, 0, -1, 0, 5, 1, -1, -1}, {true, false, true, false, false, false, true, true},
             {-1, -1, 0, 0, 3, -1, 3, -1}, {true, true, false, false, false, true, false, true},
             {10, 80, 10, 10, 10, 10, 10, 20}, {false, false, false, false, false, false, false, false},
             {-1, -1, -1, 0, 1, 0, 5}, {true, true, true, false, false, false, false},
             {-1, 0, 3, -1, -1, 0, 3}, {true, false, false, true, true, false, false},
             {15, 10, 10, 80, 10, 10, 10}, {false, false, false, false, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullKeyCount)
{
    TestGBMKCount({-1, 0, -1, 0, 5, 1, -1, -1}, {true, false, true, false, false, false, true, true},
                  {-1, -1, 0, 0, 3, -1, 3, -1}, {true, true, false, false, false, true, false, true},
                  {10, 80, 10, 10, 10, 10, 10, 20},
                  {false, false, false, false, false, false, false, false}, {-1, -1, -1, 0, 1, 0, 5},
                  {true, true, true, false, false, false, false}, {-1, 0, 3, -1, -1, 0, 3},
                  {true, false, false, true, true, false, false}, {2, 1, 1, 1, 1, 1, 1});
}

// NULL values tests
TEST(DispatcherNullTests, GroupByMultiKeyNullValueSum)
{
    TestGBMK("SUM", {0, -1, 1, -1, 2, 2, 1, 0, -1, 2, 1, 2, -1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {0, -1, -1, 1, 0, 0, 7, 0, -1, 0, 7, 0, 1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {2, -21, -1, -1, 2, -1, -1, 2, 5, -1, 1, 7, -1, 2},
             {false, false, true, false, false, true, true, false, false, true, false, false, false, false},
             {0, -1, 1, -1, 2, 1}, {false, false, false, false, false, false}, {0, -1, -1, 1, 0, 7},
             {false, false, false, false, false, false}, {6, -16, -1, -2, 9, 1},
             {false, false, true, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullValueMin)
{
    TestGBMK("MIN", {0, -1, 1, -1, 2, 2, 1, 0, -1, 2, 1, 2, -1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {0, -1, -1, 1, 0, 0, 7, 0, -1, 0, 7, 0, 1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {2, -21, -1, -1, 2, -1, -1, 2, 5, -1, 1, 7, -1, 2},
             {false, false, true, false, false, true, true, false, false, true, false, false, false, false},
             {0, -1, 1, -1, 2, 1}, {false, false, false, false, false, false}, {0, -1, -1, 1, 0, 7},
             {false, false, false, false, false, false}, {2, -21, -1, -1, 2, 1},
             {false, false, true, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullValueMax)
{
    TestGBMK("MAX", {0, -1, 1, -1, 2, 2, 1, 0, -1, 2, 1, 2, -1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {0, -1, -1, 1, 0, 0, 7, 0, -1, 0, 7, 0, 1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {2, -21, -1, -1, 2, -1, -1, 2, 5, -1, 1, 7, -1, 2},
             {false, false, true, false, false, true, true, false, false, true, false, false, false, false},
             {0, -1, 1, -1, 2, 1}, {false, false, false, false, false, false}, {0, -1, -1, 1, 0, 7},
             {false, false, false, false, false, false}, {2, 5, -1, -1, 7, 1},
             {false, false, true, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullValueAvg)
{
    TestGBMK("AVG", {0, -1, 1, -1, 2, 2, 1, 0, -1, 2, 1, 2, -1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {0, -1, -1, 1, 0, 0, 7, 0, -1, 0, 7, 0, 1, 0},
             {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
             {2, -21, -1, -1, 2, -1, -1, 2, 5, -1, 1, 7, -1, 2},
             {false, false, true, false, false, true, true, false, false, true, false, false, false, false},
             {0, -1, 1, -1, 2, 1}, {false, false, false, false, false, false}, {0, -1, -1, 1, 0, 7},
             {false, false, false, false, false, false}, {2, -8, -1, -1, 4, 1},
             {false, false, true, false, false, false});
}

TEST(DispatcherNullTests, GroupByMultiKeyNullValueCount)
{
    TestGBMKCount({0, -1, 1, -1, 2, 2, 1, 0, -1, 2, 1, 2, -1, 0},
                  {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
                  {0, -1, -1, 1, 0, 0, 7, 0, -1, 0, 7, 0, 1, 0},
                  {false, false, false, false, false, false, false, false, false, false, false, false, false, false},
                  {2, -21, -1, -1, 2, -1, -1, 2, 5, -1, 2, 7, -1, 2},
                  {false, false, true, false, false, true, true, false, false, true, false, false, false, false},
                  {0, -1, 1, -1, 2, 1}, {false, false, false, false, false, false},
                  {0, -1, -1, 1, 0, 7}, {false, false, false, false, false, false}, {3, 2, 0, 2, 2, 1});
}

TEST(DispatcherNullTests, OrEqualNullMaskMerging)
{
    TestOrEqualNullMaskMerging({0, 0, 0, 1, 1, 1, -1, -1, -1},
                               {false, false, false, false, false, false, true, true, true},
                               {0, 1, -1, 0, 1, -1, 0, 1, -1},
                               {false, false, true, false, false, true, false, false, true},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 7});
}

TEST(DispatcherNullTests, OrIsNullNullMaskMerging)
{
    TestOrIsNullNullMaskMerging({0, 0, 0, 1, 1, 1, -1, -1, -1},
                                {false, false, false, false, false, false, true, true, true},
                                {0, 1, -1, 0, 1, -1, 0, 1, -1},
                                {false, false, true, false, false, true, false, false, true},
                                {1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 6, 7, 8, 9});
}

TEST(DispatcherNullTests, OrIsNotNullMaskMerging)
{
    TestOrNotNullMaskMerging({0, 0, 0, 1, 1, 1, -1, -1, -1},
                             {false, false, false, false, false, false, true, true, true},
                             {0, 1, -1, 0, 1, -1, 0, 1, -1},
                             {false, false, true, false, false, true, false, false, true},
                             {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 7});
}
