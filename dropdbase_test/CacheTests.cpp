#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/GPUMemoryCache.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ColumnBase.h"

const std::string CACHE_TEST_DB_NAME = "db";

TEST(CacheTests, LRUTest)
{
    GPUMemoryCache cache(0, 128);

    std::tuple<char*, size_t, bool> result0 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("0"), 0, 32, 32, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result0));
    std::tuple<char*, size_t, bool> resultCahed0 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("0"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed0));

    std::tuple<char*, size_t, bool> result1 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("1"), 0, 32, 32, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result1));
    std::tuple<char*, size_t, bool> resultCahed1 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("1"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed1));

    std::tuple<char*, size_t, bool> result2 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("2"), 0, 32, 32, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result2));
    std::tuple<char*, size_t, bool> resultCahed2 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("2"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed2));

    std::tuple<char*, size_t, bool> result3 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("3"), 0, 32, 32, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result3));
    std::tuple<char*, size_t, bool> resultCahed3 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("3"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed3));

    // check if all columns are cached:
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "0", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "1", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "2", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "3", 0, 32, 0));

    std::tuple<char*, size_t, bool> result4 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("4"), 0, 32, 32, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result4));
    std::tuple<char*, size_t, bool> resultCahed4 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("4"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed4));

    // check if LRU works, column '0' should no longer be in cache:
    ASSERT_FALSE(cache.containsColumn(CACHE_TEST_DB_NAME, "0", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "4", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "1", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "2", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "3", 0, 32, 0));

    // make cache hit to change LRU simple order:
    std::tuple<char*, size_t, bool> resultCahed6 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("1"), 0, 32, 32, 0);
    ASSERT_TRUE(std::get<2>(resultCahed6));

    // insert block, which size is 2x oldBlock
    std::tuple<char*, size_t, bool> result5 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("5"), 0, 64, 64, 0); // make cache of the block
    ASSERT_FALSE(std::get<2>(result5));
    std::tuple<char*, size_t, bool> resultCahed5 =
        cache.getColumn<char>(CACHE_TEST_DB_NAME, std::string("5"), 0, 64, 64, 0);
    ASSERT_TRUE(std::get<2>(resultCahed5));

    // check if LRU works, columns '0', '2', '3' should no longer be in cache:
    ASSERT_FALSE(cache.containsColumn(CACHE_TEST_DB_NAME, "0", 0, 32, 0));
    ASSERT_FALSE(cache.containsColumn(CACHE_TEST_DB_NAME, "2", 0, 32, 0));
    ASSERT_FALSE(cache.containsColumn(CACHE_TEST_DB_NAME, "3", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "1", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "4", 0, 32, 0));
    ASSERT_TRUE(cache.containsColumn(CACHE_TEST_DB_NAME, "5", 0, 64, 0));
}
