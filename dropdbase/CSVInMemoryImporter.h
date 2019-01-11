#pragma once
#include <string>
class CSVInMemoryImporter
{
public:
	CSVInMemoryImporter(const std::string&, const std::string&);
	void ImportTables(std::shared_ptr<Database>& database);
};

