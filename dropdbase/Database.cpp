#include "Database.h"



Database::Database()
{
}


Database::~Database()
{
}

std::weak_ptr<Database> Database::GetDatabaseByName(const std::string & name) 
{
	return std::weak_ptr<Database>();
}

void Database::AddToInMemoryDatabaseList(Database && database)
{
}
