#include "Database.h"



Database::Database()
{
}


Database::~Database()
{
}

const std::shared_ptr<Database>& Database::GetDatabaseByName(const std::string & name)
{
	std::shared_ptr<Database> db;
	return db;
}

void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database>&& database)
{
}
