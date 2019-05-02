#include <memory>
#include <cstdlib>
#include <cstdio>

#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Compression/Compression.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"


class GPUCompressionTests : public ::testing::Test
{
protected:
	
	std::shared_ptr<Database> database;
	
	std::vector<int32_t> dataInt;
	std::vector<int32_t> dataIntLarge;
	std::vector<int64_t> dataLong;
	std::vector<float> dataFloatPositive;
	std::vector<float> dataFloatNegative;
	std::vector<float> dataFloatMixed;

	virtual void SetUp()
	{
		Context::getInstance();
		//Database::LoadDatabasesFromDisk();
		database = std::make_shared<Database>("compressionDatabase", 1024);
		Database::AddToInMemoryDatabaseList(database);

		std::unordered_map<std::string, DataType> columnsTable1;
		columnsTable1.insert({ "ColumnInt", COLUMN_INT });
		columnsTable1.insert({ "ColumnIntLarge", COLUMN_INT });
		columnsTable1.insert({ "ColumnLong", COLUMN_LONG });
		columnsTable1.insert({ "ColumnFloatPositive", COLUMN_FLOAT });
		columnsTable1.insert({ "ColumnFloatNegative", COLUMN_FLOAT });
		columnsTable1.insert({ "ColumnFloatMixed", COLUMN_FLOAT });
		database->CreateTable(columnsTable1, "compressionTable");

		
		for (int i = 0; i < database->GetBlockSize(); i++)
		{
			dataInt.push_back((int)(i - database->GetBlockSize() / 2));
			dataIntLarge.push_back((int)(i - database->GetBlockSize() / 2) * 100000000);
			dataLong.push_back((long)((i - database->GetBlockSize() / 2) * 100000));
			dataFloatPositive.push_back((float)(i / 100.0 + 10));
			dataFloatNegative.push_back(-(float)(i / 1000.0 + 10));
			dataFloatMixed.push_back((float)((i - database->GetBlockSize() / 2) / 100.0));
		}

		auto& columnInt = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnInt");
		auto& columnIntLarge = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnIntLarge");
		auto& columnLong = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnLong");
		auto& columnFloatPositive = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatPositive");
		auto& columnFloatNegative = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatNegative");
		auto& columnFloatMixed = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatMixed");

		dynamic_cast<ColumnBase<int32_t>*>(columnInt.get())->InsertData(dataInt);
		dynamic_cast<ColumnBase<int32_t>*>(columnIntLarge.get())->InsertData(dataIntLarge);
		dynamic_cast<ColumnBase<int64_t>*>(columnLong.get())->InsertData(dataLong);
		dynamic_cast<ColumnBase<float>*>(columnFloatPositive.get())->InsertData(dataFloatPositive);
		dynamic_cast<ColumnBase<float>*>(columnFloatNegative.get())->InsertData(dataFloatNegative);
		dynamic_cast<ColumnBase<float>*>(columnFloatMixed.get())->InsertData(dataFloatMixed);
	}

	virtual void TearDown()
	{
		for (auto& db : Database::GetDatabaseNames())
		{
			Database::RemoveFromInMemoryDatabaseList(db.c_str());
		}
	}
};


TEST_F(GPUCompressionTests, CompressionInt)
{
    BlockBase<int32_t>* block = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnInt").get())->GetBlocksList()[0];
	block->CompressData();

	ASSERT_EQ(1024, block->GetData()[0]);
	ASSERT_EQ(0, block->GetData()[1]);
	ASSERT_EQ(366, block->GetData()[2]);
	ASSERT_EQ(0, block->GetData()[3]);
	ASSERT_EQ(32, block->GetData()[4]);
	ASSERT_EQ(0, block->GetData()[5]);
	ASSERT_EQ(0, block->GetData()[6]);
	ASSERT_EQ(0, block->GetData()[7]);


	block->DecompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataInt[i], block->GetData()[i]);
	}
	
}


TEST_F(GPUCompressionTests, CompressionIntLarge)
{
	BlockBase<int32_t>* blockLarge = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnIntLarge").get())->GetBlocksList()[0];
	blockLarge->CompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataIntLarge[i], blockLarge->GetData()[i]);
	}
}




TEST_F(GPUCompressionTests, CompressionLong)
{
	BlockBase<int64_t>* block = dynamic_cast<ColumnBase<int64_t>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnLong").get())->GetBlocksList()[0];
	block->CompressData();
	
	ASSERT_EQ(1024, block->GetData()[0]);
	ASSERT_EQ(471, block->GetData()[1]);
	ASSERT_EQ(32, block->GetData()[2]);
	ASSERT_EQ(0, block->GetData()[3]);
	ASSERT_EQ(0, block->GetData()[4]);
	ASSERT_EQ(0, block->GetData()[5]);
	ASSERT_EQ(0, block->GetData()[6]);
	ASSERT_EQ(0, block->GetData()[7]);
	
	block->DecompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataLong[i], block->GetData()[i]);
	}
}




TEST_F(GPUCompressionTests, CompressionFloatPositive)
{
	auto& columnPositive = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatPositive");

	BlockBase<float>* blockPositive = dynamic_cast<ColumnBase<float>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatPositive").get())->GetBlocksList()[0];
	blockPositive->CompressData();

	int32_t * compressedDataPositive = reinterpret_cast<int32_t*>(blockPositive->GetData());

	ASSERT_EQ(1024, compressedDataPositive[0]);
	ASSERT_EQ(0, compressedDataPositive[1]);
	ASSERT_EQ(814, compressedDataPositive[2]);
	ASSERT_EQ(0, compressedDataPositive[3]);
	ASSERT_EQ(32, compressedDataPositive[4]);
	ASSERT_EQ(0, compressedDataPositive[5]);
	ASSERT_EQ(0, compressedDataPositive[6]);
	ASSERT_EQ(0, compressedDataPositive[7]);


	blockPositive->DecompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataFloatPositive[i], blockPositive->GetData()[i]);
	}
}
	

TEST_F(GPUCompressionTests, CompressionFloatNegative)
{
	auto& columnNegative = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatNegative");
	
	BlockBase<float>* blockNegative = dynamic_cast<ColumnBase<float>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatNegative").get())->GetBlocksList()[0];
	blockNegative->CompressData();

	int32_t * compressedDataNegative = reinterpret_cast<int32_t*>(blockNegative->GetData());

	ASSERT_EQ(1024, compressedDataNegative[0]);
	ASSERT_EQ(0, compressedDataNegative[1]);
	ASSERT_EQ(718, compressedDataNegative[2]);
	ASSERT_EQ(0, compressedDataNegative[3]);
	ASSERT_EQ(32, compressedDataNegative[4]);
	ASSERT_EQ(0, compressedDataNegative[5]);
	ASSERT_EQ(0, compressedDataNegative[6]);
	ASSERT_EQ(0, compressedDataNegative[7]);


	blockNegative->DecompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataFloatNegative[i], blockNegative->GetData()[i]);
	}

	// testing floats mixed

	BlockBase<float>* blockMixed = dynamic_cast<ColumnBase<float>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatMixed").get())->GetBlocksList()[0];
	blockMixed->CompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataFloatMixed[i], blockMixed->GetData()[i]);
	}
}

TEST_F(GPUCompressionTests, CompressionFloatMixed)
{
	auto& columnMixed = database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatMixed");

	BlockBase<float>* blockMixed = dynamic_cast<ColumnBase<float>*>(database->GetTables().find("compressionTable")->second.GetColumns().at("ColumnFloatMixed").get())->GetBlocksList()[0];
	blockMixed->CompressData();

	for (int i = 0; i < database->GetBlockSize(); i++) {
		ASSERT_EQ(dataFloatMixed[i], blockMixed->GetData()[i]);
	}
}






TEST_F(GPUCompressionTests, CompressionDispatcherLoadInt)
{

	GpuSqlCustomParser parser(database, "SELECT ColumnInt FROM compressionTable;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("compressionTable.ColumnInt");

	ASSERT_EQ(payloads.intpayload().intdata_size(), dataInt.size());

	for (int i = 0; i < payloads.intpayload().intdata_size(); i++)
	{
		ASSERT_EQ(dataInt[i], payloads.intpayload().intdata()[i]);
	}
}


TEST_F(GPUCompressionTests, CompressionDispatcherLoadLong)
{

	GpuSqlCustomParser parser(database, "SELECT ColumnLong FROM compressionTable;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("compressionTable.ColumnLong");

	ASSERT_EQ(payloads.int64payload().int64data_size(), dataLong.size());

	for (int i = 0; i < payloads.int64payload().int64data_size(); i++)
	{
		ASSERT_EQ(dataLong[i], payloads.int64payload().int64data()[i]);
	}
}


TEST_F(GPUCompressionTests, CompressionDispatcherLoadFloat)
{
	
	GpuSqlCustomParser parser(database, "SELECT ColumnFloatPositive FROM compressionTable;");
	auto resultPtr = parser.parse();
	auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	auto &payloads = result->payloads().at("compressionTable.ColumnFloatPositive");

	ASSERT_EQ(payloads.floatpayload().floatdata_size(), dataFloatPositive.size());

	for (int i = 0; i < payloads.floatpayload().floatdata_size(); i++)
	{
		ASSERT_FLOAT_EQ(dataFloatPositive[i], payloads.floatpayload().floatdata()[i]);
	}
}