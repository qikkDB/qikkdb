#pragma once
#include <string>
#include <memory>
#include <optional>

class Database
{
public:
	Database();
	Database(const std::string& name, int64_t blockSize) {};
	~Database();
	static std::weak_ptr<Database> GetDatabaseByName(const std::string& name);
	static void AddToInMemoryDatabaseList(const std::shared_ptr<Database>& database);
};

