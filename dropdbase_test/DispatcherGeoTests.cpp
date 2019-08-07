#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
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
	const std::vector<std::string> testPolygons = {
		"POLYGON((4.0000 4.0000, 12.0000 4.0000, 16.0000 16.0000, 4.0000 12.0000, 4.0000 4.0000), (5.0000 5.0000, 7.0000 5.0000, 7.0000 7.0000, 5.0000 5.0000))",
		"POLYGON((-7.0000 -7.0000, -0.6000 3.2000, -9.9900 89.5000, -7.0000 -7.0000), (3.2000 4.5000, 2.6789 4.2000, 150.1305 4.1000, 10.5000 2.1000, 0.6000 2.5000, 3.2000 4.5000))"
	};

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
		Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
	}

	// This is testing queries like "SELECT colID FROM SimpleTable WHERE POLYGON(...) CONTAINS colPoint;"
	// polygon - const, wkt from query_; point - col (as vector of NativeGeoPoints here)
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

		// Execute the query_
		GpuSqlCustomParser parser(geoDatabase, "SELECT colID FROM " + tableName + " WHERE GEO_CONTAINS(" + polygon + ", colPoint);");
		auto resultPtr = parser.Parse();
		auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		auto &payloads = result->payloads().at("SimpleTable.colID");

		ASSERT_EQ(expectedResult.size(), payloads.intpayload().intdata_size()) << "size is not correct";
		for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
		{
			ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
		}
	}


	void GeoContainsConstPolyConstPointTest(const std::string& polygon,
		const std::string& point,
		std::vector<int32_t> expectedResult)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		geoDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < expectedResult.size(); i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(geoDatabase->GetTables().at("SimpleTable").
			GetColumns().at("colID").get())->InsertData(colID);

		// Execute the query_
		GpuSqlCustomParser parser(geoDatabase, "SELECT colID FROM " + tableName + " WHERE GEO_CONTAINS("+ polygon +" , " + point + ");");
		auto resultPtr = parser.Parse();
		auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		auto &payloads = result->payloads().at("SimpleTable.colID");

		ASSERT_EQ(expectedResult.size(), payloads.intpayload().intdata_size()) << "size is not correct";
		for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
		{
			ASSERT_EQ(expectedResult[i], payloads.intpayload().intdata()[i]);
		}
	}

	void GeoContainsNotConstPolyConstPointTest(const std::string& polygon,
		const std::string& point)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		geoDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < 10; i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(geoDatabase->GetTables().at("SimpleTable").
			GetColumns().at("colID").get())->InsertData(colID);

		// Execute the query_
		GpuSqlCustomParser parser(geoDatabase, "SELECT colID FROM " + tableName + " WHERE GEO_CONTAINS(" + polygon + " , " + point + ");");
		auto resultPtr = parser.Parse();
		auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());
		ASSERT_EQ(result->payloads().size(), 0);
	}

	void PolygonReconstruct(std::vector<std::string> inputWkt,
		int32_t whereThreshold,
		std::vector<std::string> expectedResult)
	{
		auto columns = std::unordered_map<std::string, DataType>();
		columns.insert(std::make_pair<std::string, DataType>("colID", DataType::COLUMN_INT));
		columns.insert(std::make_pair<std::string, DataType>("colPolygon", DataType::COLUMN_POLYGON));
		geoDatabase->CreateTable(columns, tableName.c_str());

		// Create column with IDs
		std::vector<int32_t> colID;
		for (int i = 0; i < inputWkt.size(); i++)
		{
			colID.push_back(i);
		}
		reinterpret_cast<ColumnBase<int32_t>*>(geoDatabase->GetTables().at(tableName.c_str()).
			GetColumns().at("colID").get())->InsertData(colID);

		// Create column with polygons
		std::vector<ColmnarDB::Types::ComplexPolygon> colPolygon;
		for (auto wkt : inputWkt)
		{
			colPolygon.push_back(ComplexPolygonFactory::FromWkt(wkt));
		}
		reinterpret_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(geoDatabase->GetTables().at("SimpleTable").
			GetColumns().at("colPolygon").get())->InsertData(colPolygon);

		// Execute the query_
		GpuSqlCustomParser parser(geoDatabase, "SELECT colPolygon FROM " + tableName + " WHERE colID >= " +
			std::to_string(whereThreshold) + ";");
		auto resultPtr = parser.Parse();
		auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

		if(expectedResult.size() > 0)
		{
			auto &payloads = result->payloads().at("SimpleTable.colPolygon");
			ASSERT_EQ(expectedResult.size(), payloads.stringpayload().stringdata_size()) << "size is not correct";
			for (int i = 0; i < payloads.stringpayload().stringdata_size(); i++)
			{
				ASSERT_EQ(expectedResult[i], payloads.stringpayload().stringdata()[i]);
			}
		}
		else
		{
			ASSERT_EQ(result->payloads().size(), 0);
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


TEST_F(DispatcherGeoTests, GeoConstPolyConstPoint)
{
	GeoContainsConstPolyConstPointTest("POLYGON((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 5))",
		 "POINT(2 1)", 
		{ 0, 1, 2, 3, 4 });
}

TEST_F(DispatcherGeoTests, GeoNotConstPolyConstPoint)
{
	GeoContainsNotConstPolyConstPointTest("POLYGON((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 5))",
		"POINT(-2 1)");
}

TEST_F(DispatcherGeoTests, PolygonReconstructEmptyMask)
{
	PolygonReconstruct(
		testPolygons,
		2,
		{});
}

TEST_F(DispatcherGeoTests, PolygonReconstructHalfMask)
{
	PolygonReconstruct(
		testPolygons,
		1,
		{ testPolygons[1] });
}

TEST_F(DispatcherGeoTests, PolygonReconstructFullMask)
{
	PolygonReconstruct(
		testPolygons,
		0,
		testPolygons);
}
