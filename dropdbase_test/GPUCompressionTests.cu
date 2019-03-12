#include <memory>
#include <cstdlib>
#include <cstdio>

#include "gtest/gtest.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/Compression/Compression.h"



class GPUCompressionTests : public ::testing::Test
{
protected:
	std::shared_ptr<Database> compressionDatabase;

	virtual void SetUp()
	{
		Context::getInstance();
		Database::LoadDatabasesFromDisk();
	}

	virtual void TearDown()
	{
		for (auto& db : Database::GetDatabaseNames())
		{
			Database::DestroyDatabase(db.c_str());
		}
	}
};

TEST_F(GPUCompressionTests, SipmleTest)
{
	BlockBase<int32_t>* block = dynamic_cast<ColumnBase<int32_t>*>(Database::GetDatabaseByName("TestDatabase")->GetTables().find("TestTable1")->second.GetColumns().at("colInteger").get())->GetBlocksList().front().get();

	std::unique_ptr<BlockBase<int32_t>> block2 = Compression::CompressBlock<int32_t>((*block));

	for (int i = 0; i < 10; i++) {
		ASSERT_EQ(block2.get()->GetData()[i], block->GetData()[i]);
	}
}
