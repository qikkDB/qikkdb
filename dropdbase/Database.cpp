#include "Database.h"



Database::Database()
{
}


Database::~Database()
{
}

std::shared_ptr<Database> Database::GetDatabaseByName(const std::string & name)
{
	return std::shared_ptr<Database>();
}

void Database::AddToInMemoryDatabaseList(std::shared_ptr<Database>& database)
{
}
