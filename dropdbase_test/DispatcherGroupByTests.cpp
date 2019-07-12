#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "gtest/gtest.h"

// This test is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS colPoint;"

class DispatcherGroupByTests : public ::testing::Test
{
protected:
    const std::string dbName = "GroupByTestDb";
    const std::string tableName = "SimpleTable";
    const int32_t blockSize = 4; // length of a block

    std::shared_ptr<Database> groupByDatabase;

    virtual void SetUp()
    {
        Context::getInstance();

        groupByDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
        Database::AddToInMemoryDatabaseList(groupByDatabase);
    }

    virtual void TearDown()
    {
        // clean up occurs when test completes or an exception is thrown
        Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    }

    // This is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS
    // colPoint;" polygon - const, wkt from query; point - col (as vector of NativeGeoPoints here)
    void GBSGenericTest(std::string aggregationFunction,
                        std::vector<std::string> keys,
                        std::vector<int32_t> values,
                        std::unordered_map<std::string, int32_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        // Create column with IDs
        std::vector<int32_t> colID;
        for (int i = 0; i < keys.size(); i++)
        {
            colID.push_back(i);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colID").get())
            ->InsertData(colID);
        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query
        GpuSqlCustomParser parser(groupByDatabase, 
            "SELECT colString, " + aggregationFunction + "(colInteger) FROM " + tableName + " GROUP BY colString;");
        auto resultPtr = parser.parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at(aggregationFunction + "(colInteger)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.intpayload().intdata()[i])
                << " at key \"" << key << "\"";
        }
    }

	void GBOBKeysGenericTest(std::string aggregationFunction,
		std::vector<std::string> keys,
		std::vector<int32_t> values,
		std::unordered_map<int32_t, int32_t> expectedResult)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
		columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
		groupByDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < keys.size(); i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colID").get())
			->InsertData(colID);
		reinterpret_cast<ColumnBase<std::string>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
			->InsertData(keys);
		reinterpret_cast<ColumnBase<int32_t>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
			->InsertData(values);

		// Execute the query
		GpuSqlCustomParser parser(groupByDatabase,
			"SELECT colInteger, " + aggregationFunction + "(colID) FROM " + tableName + " GROUP BY colInteger ORDER BY colInteger;");
		auto resultPtr = parser.parse();
		auto result =
			dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		auto& payloadKeys = result->payloads().at(tableName + ".colInteger");
		auto& payloadValues = result->payloads().at(aggregationFunction + "(colID)");

		ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
			<< " wrong number of keys";
		for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
		{
			int32_t key = payloadKeys.intpayload().intdata()[i];
			ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
			ASSERT_EQ(expectedResult.at(key), payloadValues.intpayload().intdata()[i])
				<< " at key \"" << key << "\"";
		}
	}

	void GBOBValuesGenericTest(std::string aggregationFunction,
		std::vector<std::string> keys,
		std::vector<int32_t> values,
		std::unordered_map<int32_t, int32_t> expectedResult)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
		columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
		groupByDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < keys.size(); i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colID").get())
			->InsertData(colID);
		reinterpret_cast<ColumnBase<std::string>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
			->InsertData(keys);
		reinterpret_cast<ColumnBase<int32_t>*>(
			groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
			->InsertData(values);

		// Execute the query
		GpuSqlCustomParser parser(groupByDatabase,
			"SELECT colInteger, " + aggregationFunction + "(colID) FROM " + tableName + " GROUP BY colInteger ORDER BY " + aggregationFunction  + "(colID) - 2;");
		auto resultPtr = parser.parse();
		auto result =
			dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		auto& payloadKeys = result->payloads().at(tableName + ".colInteger");
		auto& payloadValues = result->payloads().at(aggregationFunction + "(colID)");

		ASSERT_EQ(expectedResult.size(), payloadKeys.intpayload().intdata_size())
			<< " wrong number of keys";
		for (int32_t i = 0; i < payloadKeys.intpayload().intdata_size(); i++)
		{
			int32_t key = payloadKeys.intpayload().intdata()[i];
			ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
			ASSERT_EQ(expectedResult.at(key), payloadValues.intpayload().intdata()[i])
				<< " at key \"" << key << "\"";
		}
	}

    void GBSCountTest(std::vector<std::string> keys,
                        std::vector<int32_t> values,
                        std::unordered_map<std::string, int64_t> expectedResult)
    {
        auto columns = std::unordered_map<std::string, DataType>();
        columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
        columns.insert(std::make_pair<std::string, DataType>("colString", DataType::COLUMN_STRING));
        columns.insert(std::make_pair<std::string, DataType>("colInteger", DataType::COLUMN_INT));
        groupByDatabase->CreateTable(columns, tableName.c_str());

        // Create column with IDs
        std::vector<int32_t> colID;
        for (int i = 0; i < keys.size(); i++)
        {
            colID.push_back(i);
        }
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colID").get())
            ->InsertData(colID);
        reinterpret_cast<ColumnBase<std::string>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colString").get())
            ->InsertData(keys);
        reinterpret_cast<ColumnBase<int32_t>*>(
            groupByDatabase->GetTables().at(tableName).GetColumns().at("colInteger").get())
            ->InsertData(values);

        // Execute the query
        GpuSqlCustomParser parser(groupByDatabase, 
            "SELECT colString, COUNT(colInteger) FROM " + tableName + " GROUP BY colString;");
        auto resultPtr = parser.parse();
        auto result =
            dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
        auto& payloadKeys = result->payloads().at(tableName + ".colString");
        auto& payloadValues = result->payloads().at("COUNT(colInteger)");

        ASSERT_EQ(expectedResult.size(), payloadKeys.stringpayload().stringdata_size())
            << " wrong number of keys";
        for (int32_t i = 0; i < payloadKeys.stringpayload().stringdata_size(); i++)
        {
            std::string key = payloadKeys.stringpayload().stringdata()[i];
            ASSERT_FALSE(expectedResult.find(key) == expectedResult.end()) << " key \"" << key << "\"";
            ASSERT_EQ(expectedResult[key], payloadValues.int64payload().int64data()[i])
                << " at key \"" << key << "\"";
        }
    }
};

TEST_F(DispatcherGroupByTests, StringSimpleSum)
{
    GBSGenericTest("SUM",
                   {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {      1,      2,       3,     4,        5,     6,      7,  10,    13,    15},
                   {{"Apple", 4}, {"Abcd", 9}, {"Banana", 5}, {"XYZ", 38}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleMin)
{
    GBSGenericTest("MIN",
                   {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {      1,      2,       3,     4,        5,     6,      7,  10,    13,    15},
                   {{"Apple", 1}, {"Abcd", 2}, {"Banana", 5}, {"XYZ", 4}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleMax)
{
    GBSGenericTest("MAX",
                   {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {      1,      2,       3,     4,        5,     6,      7,  10,    13,    15},
                   {{"Apple", 3}, {"Abcd", 7}, {"Banana", 5}, {"XYZ", 15}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleaAvg)
{
    GBSGenericTest("AVG",
                   {"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {      1,      2,       3,     4,        5,     6,      7,  10,    13,    15},
                   {{"Apple", 2}, {"Abcd", 4}, {"Banana", 5}, {"XYZ", 9}, {"0", 10}});
}

TEST_F(DispatcherGroupByTests, StringSimpleCount)
{
    GBSCountTest({"Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ"},
                   {      1,      2,       3,     4,        5,     6,      7,  10,    13,    15},
                   {{"Apple", 2}, {"Abcd", 2}, {"Banana", 1}, {"XYZ", 4}, {"0", 1}});
}

TEST_F(DispatcherGroupByTests, IntegerSimpleSumOrderByKeys)
{
	GBOBKeysGenericTest("SUM",
		{ "Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ" },
		{ 10,	10,	  2,	 2,		6,	   6,   4,   4,   8,     8, },
		{ {2, 5}, {4, 13}, {6, 9}, {8, 17}, {10, 1} });
}

TEST_F(DispatcherGroupByTests, IntegerSimpleSumOrderByValues)
{
	GBOBValuesGenericTest("SUM",
		{ "Apple", "Abcd", "Apple", "XYZ", "Banana", "XYZ", "Abcd", "0", "XYZ", "XYZ" },
		{ 10,	10,	  2,	 2,		6,	   6,   4,   4,   8,     8, },
		{ {10, 1}, {2, 5}, {6, 9}, {4, 13}, {8, 17} });
}


