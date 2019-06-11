#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"

TEST(ColumnTests, AddBlockWithData)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt({ 1024 });
	std::vector<int64_t> dataLong({ 1000000000000000000 });
	std::vector<float> dataFloat({ (float) 0.1111 });
	std::vector<double> dataDouble({ 0.1111111 });
	std::vector<ColmnarDB::Types::Point> dataPoint({ PointFactory::FromWkt("POINT(10.11 11.1)") });
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon({ ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") });
	std::vector<std::string> dataString({ "randomString" });

	auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock(dataInt);
	auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock(dataLong);
	auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock(dataFloat);
	auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock(dataDouble);
	auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock(dataPoint);
	auto& blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock(dataPolygon);
	auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock(dataString);

	ASSERT_EQ(blockInt.GetData()[0], dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList().front()->GetData()[0]);
	ASSERT_EQ(blockLong.GetData()[0], dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList().front()->GetData()[0]);
	ASSERT_EQ(blockFloat.GetData()[0], dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList().front()->GetData()[0]);
	ASSERT_EQ(blockDouble.GetData()[0], dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList().front()->GetData()[0]);
	ASSERT_EQ(PointFactory::WktFromPoint(blockPoint.GetData()[0]), PointFactory::WktFromPoint(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList().front()->GetData()[0]));
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(blockPolygon.GetData()[0]), ComplexPolygonFactory::WktFromPolygon(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList().front()->GetData()[0]));
	ASSERT_EQ(blockString.GetData()[0], dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList().front()->GetData()[0]);
}

TEST(ColumnTests, AddBlockWithoutData)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	auto& blockInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->AddBlock();
	auto& blockLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->AddBlock();
	auto& blockFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->AddBlock();
	auto& blockDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->AddBlock();
	auto& blockPoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->AddBlock();
	auto& blockPolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->AddBlock();
	auto& blockString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->AddBlock();

	ASSERT_EQ(blockInt.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockInt.EmptyBlockSpace(), dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockLong.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockLong.EmptyBlockSpace(), dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockFloat.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockFloat.EmptyBlockSpace(), dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockDouble.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockDouble.EmptyBlockSpace(), dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockPoint.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockPoint.EmptyBlockSpace(), dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockPolygon.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockPolygon.EmptyBlockSpace(), dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList()[0]->EmptyBlockSpace());

	ASSERT_EQ(blockString.EmptyBlockSpace(), 1024);
	ASSERT_EQ(blockString.EmptyBlockSpace(), dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList()[0]->EmptyBlockSpace());
}

TEST(ColumnTests, InsertData_BlocksDoNotExist)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloat;
	std::vector<double> dataDouble;
	std::vector<ColmnarDB::Types::Point> dataPoint;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
	std::vector<std::string> dataString;

	for (int i = 0; i < database->GetBlockSize() / 2; i++)
	{
		dataInt.push_back(i);
		dataLong.push_back(i * 1000000000);
		dataFloat.push_back((float) 0.1111 * i);
		dataDouble.push_back((double)i);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
		dataString.push_back("abc");
	}

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	std::vector<int32_t> dataInIntBlock;
	std::vector<int64_t> dataInLongBlock;
	std::vector<float> dataInFloatBlock;
	std::vector<double> dataInDoubleBlock;
	std::vector<ColmnarDB::Types::Point> dataInPointBlock;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataInPolygonBlock;
	std::vector<std::string> dataInStringBlock;

	for (auto &block : dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInIntBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInLongBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInFloatBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInDoubleBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPointBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPolygonBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInStringBlock.push_back(block->GetData()[i]);
		}
	}


	for (int i = 0; i < database->GetBlockSize() / 2; i++)
	{
		ASSERT_EQ(dataInt[i], dataInIntBlock[i]);
		ASSERT_EQ(dataLong[i], dataInLongBlock[i]);
		ASSERT_EQ(dataFloat[i], dataInFloatBlock[i]);
		ASSERT_EQ(dataDouble[i], dataInDoubleBlock[i]);
		ASSERT_EQ(PointFactory::WktFromPoint(dataPoint[i]), PointFactory::WktFromPoint(dataInPointBlock[i]));
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dataPolygon[i]), ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[i]));
		ASSERT_EQ(dataString[i], dataInStringBlock[i]);
	}
}

TEST(ColumnTests, InsertData_BlocksExistAndNewDataFitIntoThem)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloat;
	std::vector<double> dataDouble;
	std::vector<ColmnarDB::Types::Point> dataPoint;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
	std::vector<std::string> dataString;

	for (int i = 0; i < database->GetBlockSize() / 4; i++)
	{
		dataInt.push_back(i);
		dataLong.push_back(i * 1000000000);
		dataFloat.push_back((float) 0.1111 * i);
		dataDouble.push_back((double)i);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
		dataString.push_back("abc");
	}

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	std::vector<int32_t> dataInIntBlock;
	std::vector<int64_t> dataInLongBlock;
	std::vector<float> dataInFloatBlock;
	std::vector<double> dataInDoubleBlock;
	std::vector<ColmnarDB::Types::Point> dataInPointBlock;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataInPolygonBlock;
	std::vector<std::string> dataInStringBlock;

	for (auto &block : dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInIntBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInLongBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInFloatBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInDoubleBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPointBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPolygonBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInStringBlock.push_back(block->GetData()[i]);
		}
	}


	for (int i = 0; i < database->GetBlockSize() / 2; i++)
	{
		ASSERT_EQ(dataInt[i % (database->GetBlockSize() / 4)], dataInIntBlock[i]);
		ASSERT_EQ(dataLong[i % (database->GetBlockSize() / 4)], dataInLongBlock[i]);
		ASSERT_EQ(dataFloat[i % (database->GetBlockSize() / 4)], dataInFloatBlock[i]);
		ASSERT_EQ(dataDouble[i % (database->GetBlockSize() / 4)], dataInDoubleBlock[i]);
		ASSERT_EQ(PointFactory::WktFromPoint(dataPoint[i % (database->GetBlockSize() / 4)]), PointFactory::WktFromPoint(dataInPointBlock[i]));
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dataPolygon[i % (database->GetBlockSize() / 4)]), ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[i]));
		ASSERT_EQ(dataString[i % (database->GetBlockSize() / 4)], dataInStringBlock[i]);
	}
}

TEST(ColumnTests, InsertData_BlocksExistAndNewDataDoNotFitIntoThem)
{
	auto database = std::make_shared<Database>("testDatabase", 256);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloat;
	std::vector<double> dataDouble;
	std::vector<ColmnarDB::Types::Point> dataPoint;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
	std::vector<std::string> dataString;

	for (int i = 0; i < database->GetBlockSize() * 1.5; i++)
	{
		dataInt.push_back(i);
		dataLong.push_back(i * 1000000000);
		dataFloat.push_back((float) 0.1111 * i);
		dataDouble.push_back((double)i);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))"));
		dataString.push_back("abc");
	}

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	std::vector<int32_t> dataInIntBlock;
	std::vector<int64_t> dataInLongBlock;
	std::vector<float> dataInFloatBlock;
	std::vector<double> dataInDoubleBlock;
	std::vector<ColmnarDB::Types::Point> dataInPointBlock;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataInPolygonBlock;
	std::vector<std::string> dataInStringBlock;

	for (auto &block : dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInIntBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInLongBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInFloatBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInDoubleBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPointBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPolygonBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInStringBlock.push_back(block->GetData()[i]);
		}
	}


	for (int i = 0; i < database->GetBlockSize() * 1.5; i++)
	{
		ASSERT_EQ(dataInt[i], dataInIntBlock[i]);
		ASSERT_EQ(dataLong[i], dataInLongBlock[i]);
		ASSERT_EQ(dataFloat[i], dataInFloatBlock[i]);
		ASSERT_EQ(dataDouble[i], dataInDoubleBlock[i]);
		ASSERT_EQ(PointFactory::WktFromPoint(dataPoint[i]), PointFactory::WktFromPoint(dataInPointBlock[i]));
		ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dataPolygon[i]), ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[i]));
		ASSERT_EQ(dataString[i], dataInStringBlock[i]);
	}
}

TEST(ColumnTests, GetUniqueBuckets)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt({ { 1024 }, { 256 }, { 512 }, { 1024 }, { 512 } });
	std::vector<int64_t> dataLong({ { 1000000000000000000 }, {1000000000000000001}, {1000000000000000000} });
	std::vector<float> dataFloat({ { (float) 0.1111 },{ (float) 0.1112 },{ (float) 0.1113 },{ (float) 0.1114 },{ (float) 0.1111 } });
	std::vector<double> dataDouble({ { 0.1111111 }, { 0.1111116 }, { 0.1111111 } });
	std::vector<ColmnarDB::Types::Point> dataPoint({ { PointFactory::FromWkt("POINT(10.11 11.1)") },{ PointFactory::FromWkt("POINT(10.11 11.1)") } });
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon({ ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11),(21 30, 35.55 36, 30.11 20.26, 21 30),(61 80.11,90 89.15,112.12 110, 61 80.11))") });
	std::vector<std::string> dataString({ { "randomString" }, { "abc" } });

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	auto uniqueInt = (dynamic_cast<ColumnBase<int32_t>*>(columnInt.get()))->GetUniqueBuckets();
	auto uniqueLong = (dynamic_cast<ColumnBase<int64_t>*>(columnLong.get()))->GetUniqueBuckets();
	auto uniqueFloat = (dynamic_cast<ColumnBase<float>*>(columnFloat.get()))->GetUniqueBuckets();
	auto uniqueDouble = (dynamic_cast<ColumnBase<double>*>(columnDouble.get()))->GetUniqueBuckets();
	auto uniquePoint = (dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get()))->GetUniqueBuckets();
	auto uniquePolygon = (dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get()))->GetUniqueBuckets();
	auto uniqueString = (dynamic_cast<ColumnBase<std::string>*>(columnString.get()))->GetUniqueBuckets();

	ASSERT_EQ(uniqueInt.size(), 3);
	ASSERT_EQ(uniqueLong.size(), 2);
	ASSERT_EQ(uniqueFloat.size(), 4);
	ASSERT_EQ(uniqueDouble.size(), 2);
	ASSERT_EQ(uniquePoint.size(), 1);
	ASSERT_EQ(uniquePolygon.size(), 1);
	ASSERT_EQ(uniqueString.size(), 2);
}

TEST(ColumnTests, InsertNull)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertNullData(512);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertNullData(512);

	ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList()[0]->GetSize(),512);
	ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList()[0]->GetSize(),512);

	std::vector<int32_t> dataInIntBlock;
	std::vector<int64_t> dataInLongBlock;
	std::vector<float> dataInFloatBlock;
	std::vector<double> dataInDoubleBlock;
	std::vector<ColmnarDB::Types::Point> dataInPointBlock;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataInPolygonBlock;
	std::vector<std::string> dataInStringBlock;

	for (auto &block : dynamic_cast<ColumnBase<int32_t>*>(table.GetColumns().at("ColumnInt").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInIntBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<int64_t>*>(table.GetColumns().at("ColumnLong").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInLongBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<float>*>(table.GetColumns().at("ColumnFloat").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInFloatBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<double>*>(table.GetColumns().at("ColumnDouble").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInDoubleBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(table.GetColumns().at("ColumnPoint").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPointBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(table.GetColumns().at("ColumnPolygon").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInPolygonBlock.push_back(block->GetData()[i]);
		}
	}

	for (auto &block : dynamic_cast<ColumnBase<std::string>*>(table.GetColumns().at("ColumnString").get())->GetBlocksList())
	{
		for (size_t i = 0; i < block->GetSize(); i++)
		{
			dataInStringBlock.push_back(block->GetData()[i]);
		}
	}


	for (int i = 0; i < 512; i++)
	{
		ASSERT_EQ(0, dataInIntBlock[i]);
		ASSERT_EQ(0, dataInLongBlock[i]);
		ASSERT_EQ(0, dataInFloatBlock[i]);
		ASSERT_EQ(0, dataInDoubleBlock[i]);
		ASSERT_EQ("POINT(0 0)" , PointFactory::WktFromPoint(dataInPointBlock[i]));
		ASSERT_EQ("POLYGON()", ComplexPolygonFactory::WktFromPolygon(dataInPolygonBlock[i]));
		ASSERT_EQ("", dataInStringBlock[i]);
	}
}

TEST(ColumnTests, GetColumnType)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	DataType typeInt = dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->GetColumnType();
	DataType typeLong = dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->GetColumnType();
	DataType typeFloat = dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetColumnType();
	DataType typeDouble = dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetColumnType();
	DataType typePoint = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->GetColumnType();
	DataType typePolygon = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->GetColumnType();
	DataType typeString = dynamic_cast<ColumnBase<std::string>*>(columnString.get())->GetColumnType();

	ASSERT_EQ(typeInt, COLUMN_INT);
	ASSERT_EQ(typeLong, COLUMN_LONG);
	ASSERT_EQ(typeFloat, COLUMN_FLOAT);
	ASSERT_EQ(typeDouble, COLUMN_DOUBLE);
	ASSERT_EQ(typePoint, COLUMN_POINT);
	ASSERT_EQ(typePolygon, COLUMN_POLYGON);
	ASSERT_EQ(typeString, COLUMN_STRING);
}

TEST(ColumnTests, ColumnStatistics)
{
	auto database = std::make_shared<Database>("testDatabase", 1024);
	Table table(database, "testTable");

	table.CreateColumn("ColumnInt", COLUMN_INT);
	table.CreateColumn("ColumnLong", COLUMN_LONG);
	table.CreateColumn("ColumnFloat", COLUMN_FLOAT);
	table.CreateColumn("ColumnDouble", COLUMN_DOUBLE);
	table.CreateColumn("ColumnPoint", COLUMN_POINT);
	table.CreateColumn("ColumnPolygon", COLUMN_POLYGON);
	table.CreateColumn("ColumnString", COLUMN_STRING);

	auto& columnInt = table.GetColumns().at("ColumnInt");
	auto& columnLong = table.GetColumns().at("ColumnLong");
	auto& columnFloat = table.GetColumns().at("ColumnFloat");
	auto& columnDouble = table.GetColumns().at("ColumnDouble");
	auto& columnPoint = table.GetColumns().at("ColumnPoint");
	auto& columnPolygon = table.GetColumns().at("ColumnPolygon");
	auto& columnString = table.GetColumns().at("ColumnString");

	std::vector<int32_t> dataInt;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloat;
	std::vector<double> dataDouble;
	std::vector<ColmnarDB::Types::Point> dataPoint;
	std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon;
	std::vector<std::string> dataString;

	for (int i = 0; i < database->GetBlockSize(); i++)
	{
		dataInt.push_back(1);
		dataLong.push_back(100000);
		dataFloat.push_back((float) 0.1111);
		dataDouble.push_back((double) 0.1111);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 80.11, 90 89.15, 112.12 110, 61 80.11))"));
		dataString.push_back("abc");
	}

	for (int i = 0; i < database->GetBlockSize(); i++)
	{
		dataInt.push_back(5);
		dataLong.push_back(500000);
		dataFloat.push_back((float) 0.5555);
		dataDouble.push_back((double) 0.5555);
		dataPoint.push_back(PointFactory::FromWkt("POINT(10.11 11.1)"));
		dataPolygon.push_back(ComplexPolygonFactory::FromWkt("POLYGON((10 11, 11.11 12.13, 10 11), (21 30, 35.55 36, 30.11 20.26, 21 30), (61 80.11, 90 89.15, 112.12 110, 61 80.11))"));
		dataString.push_back("abc");
	}

	dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
	dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
	dynamic_cast<ColumnBase<float>*>(columnFloat.get())->InsertData(dataFloat);
	dynamic_cast<ColumnBase<double>*>(columnDouble.get())->InsertData(dataDouble);
	dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->InsertData(dataPoint);
	dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->InsertData(dataPolygon);
	dynamic_cast<ColumnBase<std::string>*>(columnString.get())->InsertData(dataString);

	ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->GetMin(), 1);
	ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->GetMax(), 5);
	ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->GetSum(), database->GetBlockSize() + 5 * database->GetBlockSize());
	ASSERT_EQ(dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->GetAvg(), 3);

	ASSERT_EQ(dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->GetMin(), 100000);
	ASSERT_EQ(dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->GetMax(), 500000);
	ASSERT_EQ(dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->GetSum(), 100000 * database->GetBlockSize() + 500000 * database->GetBlockSize());
	ASSERT_EQ(dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->GetAvg(), 300000);

	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetMin(), 0.1111);
	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetMax(), 0.5555);
	ASSERT_TRUE(std::abs(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetSum() - (0.6666 * database->GetBlockSize())) < std::abs(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetSum()) / 100000.0f);
	ASSERT_TRUE(std::abs(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetAvg() - 0.3333) < std::abs(dynamic_cast<ColumnBase<float>*>(columnFloat.get())->GetAvg()) / 100000.0f);

	ASSERT_DOUBLE_EQ(dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetMin(), 0.1111);
	ASSERT_DOUBLE_EQ(dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetMax(), 0.5555);
	ASSERT_TRUE(std::abs(dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetSum() - (0.6666 * database->GetBlockSize())) < std::abs(dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetSum()) / 100000.0f);
	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<double>*>(columnDouble.get())->GetAvg(), 0.3333);

	ASSERT_EQ(PointFactory::WktFromPoint(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->GetMin()), "POINT(0 0)");
	ASSERT_EQ(PointFactory::WktFromPoint(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->GetMax()), "POINT(0 0)");
	ASSERT_EQ(PointFactory::WktFromPoint(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->GetSum()), "POINT(0 0)");
	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columnPoint.get())->GetAvg(), 0);

	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->GetMin()), "POLYGON((0 0), (0 0))");
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->GetMax()), "POLYGON((0 0), (0 0))");
	ASSERT_EQ(ComplexPolygonFactory::WktFromPolygon(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->GetSum()), "POLYGON((0 0), (0 0))");
	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columnPolygon.get())->GetAvg(), 0);

	ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(columnString.get())->GetMin(), "");
	ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(columnString.get())->GetMax(), "");
	ASSERT_EQ(dynamic_cast<ColumnBase<std::string>*>(columnString.get())->GetSum(), "");
	ASSERT_FLOAT_EQ(dynamic_cast<ColumnBase<std::string>*>(columnString.get())->GetAvg(), 0);
}