#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/PointFactory.h"

// This test is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS colPoint;"

class DispatcherGeoTests : public ::testing::Test
{
protected:
	const std::string dbName = "GeoTestDb";
	const std::string tableName = "SimpleTable";
	const int32_t blockSize = 1 << 8; //length of a block

	std::shared_ptr<Database> geoDatabase;

	virtual void SetUp()
	{
		Context::getInstance();

		geoDatabase = std::make_shared<Database>(dbName.c_str(), blockSize);
		Database::AddToInMemoryDatabaseList(geoDatabase);
	}

	virtual void TearDown()
	{
		//clean up occurs when test completes or an exception is thrown
		Database::DestroyDatabase(dbName.c_str());
	}

	// This is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS colPoint;"
	// polygon - const, wkt from query; point - col (as vector of NativeGeoPoints here)
	void GeoContainsGenericTest(const std::string& polygon,
		std::vector<NativeGeoPoint> points,
		std::vector<int32_t> expectedResult)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		columns.insert(std::make_pair<std::string, DataType>("colPoint", DataType::COLUMN_POINT));
		geoDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < points.size(); i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(geoDatabase->GetTables().at("SimpleTable").
			GetColumns().at("colID").get())->InsertData(colID);

		// Create column with points
		std::vector<ColmnarDB::Types::Point> colPoint;
		for (auto point : points)
		{
			colPoint.push_back(PointFactory::FromLatLon(point.latitude, point.longitude));
		}
		reinterpret_cast<ColumnBase<ColmnarDB::Types::Point>*>(geoDatabase->GetTables().at("SimpleTable").
			GetColumns().at("colPoint").get())->InsertData(colPoint);

		// Execute the query
		GpuSqlCustomParser parser(geoDatabase, "SELECT colID FROM " + tableName + " WHERE GEO_CONTAINS(" + polygon + ", colPoint);");
		auto resultPtr = parser.parse();
		auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		auto &payloads = result->payloads().at("SimpleTable.colID");

		ASSERT_EQ(expectedResult.size(), payloads.intpayload().intdata_size()) << "size is not correct";
		for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
		{
			ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
		}
	}
};


TEST_F(DispatcherGeoTests, GeoConvexPolygonContains)
{
	GeoContainsGenericTest("POLYGON((1.0 1.0,3.0 1.0,1.0 3.0,1.0 1.0))",
		{ {0.5, 0.5}, {2.5, 0.5}, {1.1, 1.1}, {2.8, 1.1}, {0.5, 2.0}, {1.5, 2.0}, {1.9, 2.0},
		{2.1, 2.0}, {1.01, 2.95}, {0.5, 3.0}, {1.0, 4.0}, {2.0, 3.0}, {4.0, 1.0} },
		{ 2, 3, 5, 6, 8 });
}

TEST_F(DispatcherGeoTests, GeoConcavePolygonContains)
{
	GeoContainsGenericTest("POLYGON((1.0 2.0,3.0 1.0,3.0 2.0,2.0 2.0,2.0 3.0,1.0 3.0,1.0 2.0))",
		{ {-1.0, -1.0}, {1.9, 1.5}, {0.5, 2.5}, {1.5, 3.1}, {3.5, 4.0}, {2.1, 2.1}, {3.5, 1.5},
		{3.1, 0.9}, {1.5, 2.5}, {1.9, 1.9}, {2.1, 1.9}, {2.9, 1.1} },
		{ 8, 9, 10, 11 });
}

// Very hard test - "butterfly" polygon, like |><|
TEST_F(DispatcherGeoTests, GeoTrickyPolygonContains)
{
	GeoContainsGenericTest("POLYGON((30 30,10 20,10 30,30 20,30 30))",
		{ {5, 15}, {5, 25}, {5, 35}, {16, 24}, {15, 25}, {25, 21}, {28, 25}, {30.05, 25}, {20, 25.1}, {20, 35} },
		{ 3, 4, 6 });
}

// Complex polygon (Non intersected 2 polygons)
TEST_F(DispatcherGeoTests, GeoSimpleComplexPolygonContains)
{
	GeoContainsGenericTest("POLYGON((0 0,1 0,1 1,0 1,0 0),(2 2,3 2,3 3,2 3,2 2))",
		{ {0.5, 0.5}, {1.5, 1.5}, {0.5, 2.5}, {2.5, 0.5}, {2.5, 2.5}, {50, -20} },
		{ 0, 4 });
}

// Gappy complex polygon
TEST_F(DispatcherGeoTests, GeoNestedComplexPolygonContains)
{
	GeoContainsGenericTest("POLYGON((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 5))",
		{ {-2, 1}, {1, 1}, {6, 1}, {1, 6}, {6, 5.5}, {5.5, 6}, {8, 8}, {12, 8}, {12, 8}, {20, 20} },
		{ 1, 2, 3, 5, 6 });
}
