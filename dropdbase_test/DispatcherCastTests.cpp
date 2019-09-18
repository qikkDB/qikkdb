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

    void CastStringToPointGenericTest(std::vector<std::string> strings, std::vector<std::string> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        castDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<std::string>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(strings);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colString AS GEO_POINT) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colStringASGEO_POINT)");

        ASSERT_EQ(expectedResult.size(), payloadCast.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.stringpayload().stringdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payloadCast.stringpayload().stringdata()[i]) << i;
        }
    }

    void CastPolygonToStringGenericTest(std::vector<std::string> polygonWkts, std::vector<std::string> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colPolygon", DataType::COLUMN_POLYGON));
        castDatabase->CreateTable(columns, tableName.c_str());

        std::vector<ColmnarDB::Types::ComplexPolygon> polygons;

        std::transform(polygonWkts.data(), polygonWkts.data() + polygonWkts.size(), std::back_inserter(polygons),
                       [](const std::string& polygonWkt) -> ColmnarDB::Types::ComplexPolygon {
                           return ComplexPolygonFactory::FromWkt(polygonWkt);
                       });

        reinterpret_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colPolygon").get())
            ->InsertData(polygons);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colPolygon AS STRING) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colPolygonASSTRING)");

        ASSERT_EQ(expectedResult.size(), payloadCast.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.stringpayload().stringdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payloadCast.stringpayload().stringdata()[i]) << i;
        }
    }

    void CastPointToStringGenericTest(std::vector<std::string> pointWkts, std::vector<std::string> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colPoint", DataType::COLUMN_POINT));
        castDatabase->CreateTable(columns, tableName.c_str());

        std::vector<ColmnarDB::Types::Point> points;

        std::transform(pointWkts.data(), pointWkts.data() + pointWkts.size(), std::back_inserter(points),
                       [](const std::string& pointWkt) -> ColmnarDB::Types::Point {
                           return PointFactory::FromWkt(pointWkt);
                       });

        reinterpret_cast<ColumnBase<ColmnarDB::Types::Point>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colPoint").get())
            ->InsertData(points);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colPoint AS STRING) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colPointASSTRING)");

        ASSERT_EQ(expectedResult.size(), payloadCast.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.stringpayload().stringdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payloadCast.stringpayload().stringdata()[i]) << i;
        }
    }

    void CastConstPointToStringGenericTest(std::string pointWkt, std::string expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        auto& table = castDatabase->CreateTable(columns, tableName.c_str());
        table.CreateColumn("test", COLUMN_INT);
        std::unordered_map<std::string, std::any> data;
        std::any data_vec = std::vector<int>({1, 2, 3, 4, 5});
        data.insert(std::make_pair("test", data_vec));
        table.InsertData(data);
        // Execute the query_
        GpuSqlCustomParser parser(castDatabase,
                                  "SELECT CAST(" + pointWkt + " AS STRING) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(" + pointWkt + "ASSTRING)");

        ASSERT_EQ(expectedResult, payloadCast.stringpayload().stringdata()[0]);
    }

    void CastFloatToStringGenericTest(std::vector<float> floats, std::vector<std::string> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colFloat", DataType::COLUMN_FLOAT));
        castDatabase->CreateTable(columns, tableName.c_str());

        reinterpret_cast<ColumnBase<float>*>(
            castDatabase->GetTables().at(tableName).GetColumns().at("colFloat").get())
            ->InsertData(floats);

        // Execute the query_
        GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colFloat AS STRING) FROM " + tableName + ";");
        auto resultPtr = parser.Parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadCast = result->payloads().at("CAST(colFloatASSTRING)");

        ASSERT_EQ(expectedResult.size(), payloadCast.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadCast.stringpayload().stringdata_size(); i++)
        {
            ASSERT_EQ(expectedResult[i], payloadCast.stringpayload().stringdata()[i]) << i;
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

TEST_F(DispatcherCastTests, StringToPointTest)
{
    CastStringToPointGenericTest({"POINT(12.22 12.345)", "POINT(13 13.3456)", "POINT(14.267 14)",
                                  "POINT( 15.2   15.3)", "POINT(   16.2 16.3   )", "POINT(    17.2    17.3 )",
                                  "POINT( 18.2   18.3 )", "POINT( 18.2  18.3    )"},
                                 {"POINT(12.2200 12.3450)", "POINT(13.0000 13.3456)", "POINT(14.2670 14.0000)",
                                  "POINT(15.2000 15.3000)", "POINT(16.2000 16.3000)", "POINT(17.2000 17.3000)",
                                  "POINT(18.2000 18.3000)", "POINT(18.2000 18.3000)"});
}

TEST_F(DispatcherCastTests, PolygonToStringTest)
{
    CastPolygonToStringGenericTest(
        {"POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))"},
        {"POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))",
         "POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 "
         "4.0000), "
         "(5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
         "POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 "
         "4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 "
         "4.5000))"});
}

TEST_F(DispatcherCastTests, PointToStringTest)
{
    CastPointToStringGenericTest({"POINT(12.2200 12.3450)", "POINT(13.0000 13.3456)", "POINT(14.2670 14.0000)",
                                  "POINT(15.2000 15.3000)", "POINT(16.2000 16.3000)", "POINT(17.2000 17.3000)",
                                  "POINT(18.2000 18.3000)", "POINT(18.2000 18.3000)"},
                                 {"POINT(12.2200 12.3450)", "POINT(13.0000 13.3456)", "POINT(14.2670 14.0000)",
                                  "POINT(15.2000 15.3000)", "POINT(16.2000 16.3000)", "POINT(17.2000 17.3000)",
                                  "POINT(18.2000 18.3000)", "POINT(18.2000 18.3000)"});
}

TEST_F(DispatcherCastTests, PointToStringConstTest)
{
    CastConstPointToStringGenericTest("POINT(12.2200 12.3450)", "POINT(12.2200 12.3450)");
}

TEST_F(DispatcherCastTests, FloatToStringTest)
{
    CastFloatToStringGenericTest({-1.23, 512.3, 231.0, 5.3123, 1.002, 76.9, -123.23, 123.67},
                                 {"-1.23", "512.3", "231", "5.3123", "1.002", "76.9", "-123.23", "123.67"});
}

TEST_F(DispatcherCastTests, IntToStringTest)
{
    auto columns = std::unordered_map<std::string, DataType>();
    columns.insert(std::make_pair<std::string, DataType>("colInt", DataType::COLUMN_INT));
    castDatabase->CreateTable(columns, tableName.c_str());

    reinterpret_cast<ColumnBase<float>*>(
        castDatabase->GetTables().at(tableName).GetColumns().at("colInt").get())
        ->InsertData({1, -1, 10, 123456, -1245732});

    // Execute the query_
    GpuSqlCustomParser parser(castDatabase, "SELECT CAST(colInt AS STRING) FROM " + tableName + ";");
    auto resultPtr = parser.Parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
    auto& payloadCast = result->payloads().at("CAST(colIntASSTRING)");
    std::vector<std::string> expectedResult = {"1", "-1", "10", "123456", "-1245732"};
    ASSERT_EQ(expectedResult.size(), payloadCast.stringpayload().stringdata_size())
        << " wrong number of keys";
    for (int32_t i = 0; i < payloadCast.stringpayload().stringdata_size(); i++)
    {
        ASSERT_EQ(expectedResult[i], payloadCast.stringpayload().stringdata()[i]) << i;
    }
}