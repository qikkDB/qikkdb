#include "CSVDataImporter.h"
#include <CSVParser.hpp>

void CSVDataImporter::ImportTables(std::shared_ptr<Database> database)
{
	if (this->header)
		this->ExtractHeaders();

	this->ExtractTypes();

	// prepares map columnName -> columnType
	std::unordered_map<std::string, DataType> columns;
	for (int i = 0; i < headers.size(); i++) {
		columns[headers[i]] = dataTypes[i];
	}
	
	// creates table
	Table& table = database->CreateTable(columns, this->ExtractTableNameFromFileName(fileName).c_str());
	
	// initializes map columnName -> vector of column data
	std::unordered_map<std::string, std::any> data;
	for (int i = 0; i < headers.size(); i++) {
		if (dataTypes[i] == COLUMN_INT)
		{
			std::vector<int32_t> v;
			v.reserve(database->GetBlockSize());
			data[headers[i]] = v;
		}
		else if (dataTypes[i] == COLUMN_LONG)
		{
			std::vector<int64_t> v;
			v.reserve(database->GetBlockSize());
			data[headers[i]] = v;
		}
		else if (dataTypes[i] == COLUMN_FLOAT)
		{
			std::vector<float> v;
			v.reserve(database->GetBlockSize());
			data[headers[i]] = v;
		}
		else if (dataTypes[i] == COLUMN_DOUBLE)
		{
			std::vector<double> v;
			v.reserve(database->GetBlockSize());
			data[headers[i]] = v;
		}
		else if (dataTypes[i] == COLUMN_STRING)
		{
			std::vector<std::string> v;
			v.reserve(database->GetBlockSize());
			data[headers[i]] = v;
		}
	}
	
	std::ifstream f(fileName);
	aria::csv::CsvParser parser = aria::csv::CsvParser(f).delimiter(delimiter).quote(quotes);

	int position = 0;

	// parses file and inserts data in batches of size of blockSize
	for (auto& row : parser) {

		int columnIndex = 0;
		for (auto& field : row) {
						
			if (position > 0) {
				std::any &wrappedData = data.at(headers[columnIndex]);
				
				switch (dataTypes[columnIndex])
				{
				case COLUMN_INT:
					std::any_cast<std::vector<int32_t>&>(wrappedData).push_back(std::stoi(field));
					break;
				case COLUMN_LONG:
					std::any_cast<std::vector<int64_t>&>(wrappedData).push_back(std::stol(field));
					break;
				case COLUMN_FLOAT:
					std::any_cast<std::vector<float>&>(wrappedData).push_back(std::stof(field));
					break;
				case COLUMN_DOUBLE:
					std::any_cast<std::vector<double>&>(wrappedData).push_back(std::stod(field));
					break;
				case COLUMN_STRING:
					std::any_cast<std::vector<std::string>&>(wrappedData).push_back(field);
					break;
				}
			}
			
			columnIndex++;
		}
		
		position++;
		
		// inserts parsed data into database when blockSize reached
		if (position % database->GetBlockSize() == 0) {
			
			table.InsertData(data);

			// clears parsed data so far
			for (int i = 0; i < headers.size(); i++) {
				std::any &wrappedData = data.at(headers[i]);
				switch (dataTypes[i])
				{
				case COLUMN_INT:
					std::any_cast<std::vector<int32_t>&>(wrappedData).clear();
					break;
				case COLUMN_LONG:
					std::any_cast<std::vector<int64_t>&>(wrappedData).clear();
					break;
				case COLUMN_FLOAT:
					std::any_cast<std::vector<float>&>(wrappedData).clear();
					break;
				case COLUMN_DOUBLE:
					std::any_cast<std::vector<double>&>(wrappedData).clear();
					break;
				case COLUMN_STRING:
					std::any_cast<std::vector<std::string>&>(wrappedData).clear();
					break;
				}
			}
		}
	}
}

void CSVDataImporter::ExtractHeaders()
{
	std::ifstream f(fileName);
	aria::csv::CsvParser parser = aria::csv::CsvParser(f).delimiter(delimiter).quote(quotes);

	int position = 0;
	for (auto& row : parser) {

		int columnIndex = 0;
		for (auto& field : row) {

			if (position == 0) {
				this->headers.push_back(field);
			}
			
			columnIndex++;
		}

		position++;
		if (position == 1)
			break;
	}	
}

void CSVDataImporter::ExtractTypes()
{
	std::vector<std::vector<std::string>> columnData;

	std::ifstream f(fileName);
	aria::csv::CsvParser parser = aria::csv::CsvParser(f).delimiter(delimiter).quote(quotes);

	int position = 0;

	for (auto& row : parser) {

		int columnIndex = 0;
		for (auto& field : row) {

			if (position == 0) {
				columnData.push_back(std::vector<std::string>());
			}
			else {
				columnData[columnIndex].push_back(field);

			}

			columnIndex++;
		}


		position++;

		if (position == 100)
			break;
	}

	for (auto& column : columnData) {
		DataType type = this->IndetifyDataType(column);
		dataTypes.push_back(type);
	}
}

DataType CSVDataImporter::IndetifyDataType(std::vector<std::string> columnValues)
{
	std::vector<DataType> dataTypes;

	for (auto& s : columnValues) {
		try {
			size_t position;
			std::stoi(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_INT);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}

		try {
			size_t position;
			std::stol(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_LONG);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}

		try {
			size_t position;
			std::stof(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_FLOAT);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}

		try {
			size_t position;
			std::stod(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_DOUBLE);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}

		dataTypes.push_back(COLUMN_STRING);
	}

	if (dataTypes.size() > 0) {
		bool ok = true;
		DataType type = dataTypes[0];
		for (auto& t : dataTypes) {
			if (type != t)
				ok = false;
		}

		if (ok)
			return type;
		else
			return COLUMN_STRING;
	}
	return COLUMN_STRING;
}


std::any CSVDataImporter::CastStringToDataType(std::string s, DataType dataType) {
	std::any value;
	switch (dataType) {
	case COLUMN_INT:
		try {
			value = std::stoi(s);			
		}		
		catch (std::out_of_range& e) {
			value = 0;
		}
		break;
	case COLUMN_LONG:
		try {
			value = std::stol(s);
		}
		catch (std::out_of_range& e) {
			value = 0l;
		}
		break;
	case COLUMN_FLOAT:
		try {
			value = std::stof(s);
		}
		catch (std::out_of_range& e) {
			value = 0.0f;
		}
		break;
	case COLUMN_DOUBLE:
		try {
			value = std::stod(s);
		}
		catch (std::out_of_range& e) {
			value = 0.0;
		}
		break;
	case COLUMN_STRING:
		value = s;
		break;
	}
	return value;
}

std::string CSVDataImporter::ExtractTableNameFromFileName(std::string fileName)
{
	const size_t lastSlashIndex = fileName.find_last_of("\\/");
	if (std::string::npos != lastSlashIndex)
	{
		fileName.erase(0, lastSlashIndex + 1);
	}

	const size_t extensionIdx = fileName.rfind('.');
	if (std::string::npos != extensionIdx)
	{
		fileName.erase(extensionIdx);
	}
	return fileName;
}
	
