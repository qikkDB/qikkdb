#pragma once
#include <string>
#include <memory>

class Database
{
public:
	Database();
	Database(const std::string& name, int64_t blockSize) {};
	~Database();
	static std::shared_ptr<Database> GetDatabaseByName(const std::string& name);
	static void AddToInMemoryDatabaseList(std::shared_ptr<Database>& database);
};

