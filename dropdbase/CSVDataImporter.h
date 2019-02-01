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
	CSVDataImporter(std::string fileName, bool header=true, char delimiter = ',', char quotes = '\'', char decimal = '.') {
		this->fileName = fileName;
		this->delimiter = delimiter;
		this->quotes = quotes;
		this->decimal = decimal;
		this->header = header;		
	}

	void ImportTables(std::shared_ptr<Database> database);
	void ExtractHeaders();
	void ExtractTypes();

private:
	std::string fileName;
	bool header;
	char delimiter;
	char quotes;
	char decimal;
	std::vector<std::string> headers;
	std::vector<DataType> dataTypes;	
	std::unordered_map<std::string, std::any> data;

	DataType IndetifyDataType(std::vector<std::string> columnValues);	
	std::string ExtractTableNameFromFileName(std::string fileName);
};
