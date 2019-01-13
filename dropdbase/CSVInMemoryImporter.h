#pragma once
#include <string>
#include <memory>
#include "Database.h"

class CSVInMemoryImporter
{
public:
	CSVInMemoryImporter(const std::string&, const std::string&);
	void ImportTables(std::shared_ptr<Database>& database);
};

