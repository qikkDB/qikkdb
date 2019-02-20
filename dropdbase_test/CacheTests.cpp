#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/GPUMemoryCache.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ColumnBase.h"

TEST(CacheTests, LRUTest)
{
	GPUMemoryCache cache(0, 128);

	std::tuple<char*, size_t, bool> result0 = cache.getColumn<char>(std::string("0"), 0, 32); //make cache of the block
	ASSERT_EQ(std::get<2>(result0), false);
	std::tuple<char*, size_t, bool> resultCahed0 = cache.getColumn<char>(std::string("0"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed0), true);

	std::tuple<char*, size_t, bool> result1 = cache.getColumn<char>(std::string("1"), 0, 32); //make cache of the block
	ASSERT_EQ(std::get<2>(result1), false);
	std::tuple<char*, size_t, bool> resultCahed1 = cache.getColumn<char>(std::string("1"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed1), true);

	std::tuple<char*, size_t, bool> result2 = cache.getColumn<char>(std::string("2"), 0, 32); //make cache of the block
	ASSERT_EQ(std::get<2>(result2), false);
	std::tuple<char*, size_t, bool> resultCahed2 = cache.getColumn<char>(std::string("2"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed2), true);

	std::tuple<char*, size_t, bool> result3 = cache.getColumn<char>(std::string("3"), 0, 32); //make cache of the block
	ASSERT_EQ(std::get<2>(result3), false);
	std::tuple<char*, size_t, bool> resultCahed3 = cache.getColumn<char>(std::string("3"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed3), true);

	//check if all columns are cached:
	ASSERT_EQ(cache.containsColumn("0", 0), true);
	ASSERT_EQ(cache.containsColumn("1", 0), true);
	ASSERT_EQ(cache.containsColumn("2", 0), true);
	ASSERT_EQ(cache.containsColumn("3", 0), true);

	std::tuple<char*, size_t, bool> result4 = cache.getColumn<char>(std::string("4"), 0, 32); //make cache of the block
	ASSERT_EQ(std::get<2>(result4), false);
	std::tuple<char*, size_t, bool> resultCahed4 = cache.getColumn<char>(std::string("4"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed4), true);

	//check if LRU works, column '0' should no longer be in cache:
	ASSERT_EQ(cache.containsColumn("0", 0), false);
	ASSERT_EQ(cache.containsColumn("4", 0), true);
	ASSERT_EQ(cache.containsColumn("1", 0), true);
	ASSERT_EQ(cache.containsColumn("2", 0), true);
	ASSERT_EQ(cache.containsColumn("3", 0), true);

	//make cache hit to change LRU simple order:
	std::tuple<char*, size_t, bool> resultCahed6 = cache.getColumn<char>(std::string("1"), 0, 32);
	ASSERT_EQ(std::get<2>(resultCahed6), true);

	//insert block, which size is 2x oldBlock
	std::tuple<char*, size_t, bool> result5 = cache.getColumn<char>(std::string("5"), 0, 64); //make cache of the block
	ASSERT_EQ(std::get<2>(result5), false);
	std::tuple<char*, size_t, bool> resultCahed5 = cache.getColumn<char>(std::string("5"), 0, 64);
	ASSERT_EQ(std::get<2>(resultCahed5), true);

	//check if LRU works, columns '0', '2', '3' should no longer be in cache:
	ASSERT_EQ(cache.containsColumn("0", 0), false);
	ASSERT_EQ(cache.containsColumn("2", 0), false);
	ASSERT_EQ(cache.containsColumn("3", 0), false);
	ASSERT_EQ(cache.containsColumn("1", 0), true);
	ASSERT_EQ(cache.containsColumn("4", 0), true);
	ASSERT_EQ(cache.containsColumn("5", 0), true);
}
