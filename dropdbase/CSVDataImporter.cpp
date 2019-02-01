#include <boost/log/trivial.hpp>
#include "CSVDataImporter.h"
#include "../CSVParser.hpp"

/// <summary>
/// Parses CSV file, guess types, create table (if not exists) and fills the table with parsed data
/// </summary>
/// <param name="database">Database where data will be imported</param>
void CSVDataImporter::ImportTables(std::shared_ptr<Database> database)
{
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
	std::vector<std::any> rowData;
	rowData.reserve(headers.size());

	// parses file and inserts data in batches of size of blockSize
	for (auto& row : parser) {
		
		int columnIndex = 0;
		if (position > 0 || !this->header) {
			// casts and puts data into row vector
			// if casting fails, the line is ommited
			try {
				for (auto& field : row) {

					std::any value;
					switch (dataTypes[columnIndex]) {
					case COLUMN_INT:
						value = (int32_t)std::stol(field);
						break;
					case COLUMN_LONG:
						value = (int64_t)std::stoll(field);
						break;
					case COLUMN_FLOAT:
						value = (float)std::stof(field);						
						break;
					case COLUMN_DOUBLE:
						value = (double)std::stod(field);						
						break;
					case COLUMN_STRING:
						value = field;
						break;
					}
					rowData.push_back(value);
					columnIndex++;
				}
			}
			catch (std::out_of_range& e) {
				BOOST_LOG_TRIVIAL(warning) << "Import of file " << fileName << " failed on line " << position << " (column " << columnIndex+1 << ")";
				rowData.clear(); 
			}
			catch (std::invalid_argument& e) {
				BOOST_LOG_TRIVIAL(warning) << "Import of file " << fileName << " failed on line " << position << " (column " << columnIndex+1 << ")";
				rowData.clear();
			}
		}

		// pushes values of row vector into corresponding columns
		columnIndex = 0;
		for (auto& field : rowData) {
			std::any &wrappedData = data.at(headers[columnIndex]);
			int v;
			switch (dataTypes[columnIndex])
			{
			case COLUMN_INT:
 				std::any_cast<std::vector<int32_t>&>(wrappedData).push_back(std::any_cast<int32_t>(field));
				break;
			case COLUMN_LONG:
				std::any_cast<std::vector<int64_t>&>(wrappedData).push_back(std::any_cast<int64_t>(field));
				break;
			case COLUMN_FLOAT:
				std::any_cast<std::vector<float>&>(wrappedData).push_back(std::any_cast<float>(field));
				break;
			case COLUMN_DOUBLE:
				std::any_cast<std::vector<double>&>(wrappedData).push_back(std::any_cast<double>(field));
				break;
			case COLUMN_STRING:
				std::any_cast<std::vector<std::string>&>(wrappedData).push_back(std::any_cast<std::string>(field));
				break;
			}

			columnIndex++;			
		}

		rowData.clear();
		
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
	
	// inserts remaing rows into table
	table.InsertData(data);
}

/// <summary>
/// Extracts column names from header. If there is no header, column names are created C0, C1,...
/// </summary>
void CSVDataImporter::ExtractHeaders()
{
	std::ifstream f(fileName);
	aria::csv::CsvParser parser = aria::csv::CsvParser(f).delimiter(delimiter).quote(quotes);

	int position = 0;
	for (auto& row : parser) {

		int columnIndex = 0;
		for (auto& field : row) {

			if (position == 0) {
				if (this->header)
					this->headers.push_back(field);
				else
					this->headers.push_back("C" + std::to_string(columnIndex));
			}
			
			columnIndex++;
		}

		position++;
		if (position == 1)
			break;
	}	
}

/// <summary>
/// Extracts types based on 100 leading rows
/// </summary>
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

/// <summary>
/// Identify data type based on vector of values. Returns maximum type from vector of types.
/// COLUMN_INT < COLUMN_LONG < COLUMN_FLOAT < COLUMN_DOUBLE < COULMN_STRING
/// </summary>
/// <param name="columnValues">vector of string values</param>
/// <returns>Suitable data type</returns>
DataType CSVDataImporter::IndetifyDataType(std::vector<std::string> columnValues)
{
	std::vector<DataType> dataTypes;

	for (auto& s : columnValues) {
		try {
			size_t position;
			std::stol(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_INT);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}
		catch (std::invalid_argument& e) {
		}

		try {
			size_t position;
			std::stoll(s, &position);
			if (s.length() == position) {
				dataTypes.push_back(COLUMN_LONG);
				continue;
			}
		}
		catch (std::out_of_range& e) {
		}
		catch (std::invalid_argument& e) {
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
		catch (std::invalid_argument& e) {
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
		catch (std::invalid_argument& e) {
		}


		dataTypes.push_back(COLUMN_STRING);
	}

	if (dataTypes.size() > 0) {
		DataType maxType = dataTypes[0];
		for (auto t : dataTypes) {
			if (t > maxType) {
				maxType = t;
			}
		}
		return maxType;
	}
	return COLUMN_STRING;
}

/// <summary>
/// Extract filename without extension from file path
/// </summary>
/// <param name="polygons">Imported CSV file path</param>
/// <returns>Name of file without extension</returns>
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
	
