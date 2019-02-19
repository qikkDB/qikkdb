#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <any>
#include "Database.h"
#include "DataType.h"
#include "Table.h"

class CSVDataImporter
{
public:
	CSVDataImporter(const char* inputString, const char* tableName, bool header = true, char delimiter = ',', char quotes = '\'', char decimal = '.') :
		inputStream_(std::make_unique<std::istringstream>(inputString)),
		tableName_(tableName),
		header_(header),
		delimiter_(delimiter),
		quotes_(quotes),
		decimal_(decimal)
	{
	}
	
	CSVDataImporter(const char* fileName, bool header = true, char delimiter = ',', char quotes = '\'', char decimal = '.');
	
	void ImportTables(std::shared_ptr<Database>& database);
	void ExtractHeaders();
	void ExtractTypes();

private:
	std::unique_ptr<std::istream> inputStream_;
	std::string tableName_;
	bool header_;
	char delimiter_;
	char quotes_;
	char decimal_;
	std::vector<std::string> headers_;
	std::vector<DataType> dataTypes_;

	DataType IdentifyDataType(std::vector<std::string> columnValues);
};
