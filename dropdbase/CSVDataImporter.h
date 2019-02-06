#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <any>
#include "Database.h"
#include "DataType.h"
#include "Table.h"

class CSVDataImporter
{
public:
	CSVDataImporter(std::string fileName, bool header = true, char delimiter = ',', char quotes = '\'', char decimal = '.') :
		fileName_(fileName),
		header_(header),
		delimiter_(delimiter),
		quotes_(quotes),
		decimal_(decimal)
	{
	}

	void ImportTables(std::shared_ptr<Database> database);
	void ExtractHeaders();
	void ExtractTypes();

private:
	std::string fileName_;
	bool header_;
	char delimiter_;
	char quotes_;
	char decimal_;
	std::vector<std::string> headers_;
	std::vector<DataType> dataTypes_;

	DataType IdentifyDataType(std::vector<std::string> columnValues);
};
