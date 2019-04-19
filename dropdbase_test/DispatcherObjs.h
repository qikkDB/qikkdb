#pragma once

constexpr int32_t TEST_BLOCK_COUNT = 2;		// this must be at least 2 (for testing more cases)
constexpr int32_t TEST_BLOCK_SIZE = 1 << 11;
constexpr float pi() { return 3.1415926f; }

class DispatcherObjs
{
public:
	DispatcherObjs()
	{
		tableNames = { "TableA" };
		columnTypes = { {COLUMN_INT},    {COLUMN_INT},     {COLUMN_LONG},  {COLUMN_LONG},
					   {COLUMN_LONG},  {COLUMN_FLOAT},   {COLUMN_FLOAT}, {COLUMN_DOUBLE}, {COLUMN_DOUBLE},
					   {COLUMN_POLYGON}, {COLUMN_POLYGON}, {COLUMN_POINT}, {COLUMN_STRING} };

		database = DatabaseGenerator::GenerateDatabase("TestDb", TEST_BLOCK_COUNT, TEST_BLOCK_SIZE, false, tableNames, columnTypes);
	}
	static DispatcherObjs GetInstance()
	{
		static DispatcherObjs objs;
		return objs;
	}
	std::vector<std::string> tableNames;
	std::vector<DataType> columnTypes;
	std::shared_ptr<Database> database;
};
