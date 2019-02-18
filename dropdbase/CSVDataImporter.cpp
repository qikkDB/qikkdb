#include <boost/log/trivial.hpp>
#include <boost/filesystem/path.hpp>
#include "CSVDataImporter.h"
#include "../CSVParser.hpp"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"

CSVDataImporter::CSVDataImporter(const char* fileName, bool header, char delimiter, char quotes, char decimal) :
	inputStream_(std::make_unique<std::ifstream>(fileName)),
	tableName_(boost::filesystem::path(fileName).stem().string()),
	header_(header),
	delimiter_(delimiter),
	quotes_(quotes),
	decimal_(decimal)
{
}

/// <summary>
/// Parses CSV file, guess types, create table (if not exists) and fills the table with parsed data
/// </summary>
/// <param name="database">Database where data will be imported</param>
void CSVDataImporter::ImportTables(std::shared_ptr<Database>& database)
{
	this->ExtractHeaders();
	
	this->ExtractTypes();

	inputStream_->clear();
	inputStream_->seekg(0, std::ios::beg);
	// prepares map columnName -> columnType
	std::unordered_map<std::string, DataType> columns;
	for (int i = 0; i < headers_.size(); i++) {
		columns[headers_[i]] = dataTypes_[i];
	}
	
	// creates table
	Table& table = database->CreateTable(columns, tableName_.c_str());
	
	// initializes map columnName -> vector of column data
	std::unordered_map<std::string, std::any> data;
	for (int i = 0; i < headers_.size(); i++) {
		if (dataTypes_[i] == COLUMN_INT)
		{
			std::vector<int32_t> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_LONG)
		{
			std::vector<int64_t> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_FLOAT)
		{
			std::vector<float> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_DOUBLE)
		{
			std::vector<double> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_POINT)
		{
			std::vector<ColmnarDB::Types::Point> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_POLYGON)
		{
			std::vector<ColmnarDB::Types::ComplexPolygon> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else if (dataTypes_[i] == COLUMN_STRING)
		{
			std::vector<std::string> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
		else {
			std::vector<std::string> v;
			v.reserve(database->GetBlockSize());
			data[headers_[i]] = std::move(v);
		}
	}
	
	aria::csv::CsvParser parser = aria::csv::CsvParser(*inputStream_).delimiter(delimiter_).quote(quotes_);

	int position = 0;
	std::vector<std::any> rowData;
	rowData.reserve(headers_.size());

	// parses file and inserts data in batches of size of blockSize
	for (auto& row : parser) {
		
		int columnIndex = 0;
		if (position > 0 || !this->header_) {
			// casts and puts data into row vector
			// if casting fails, the line is ommited
			try {
				for (auto& field : row) {

					std::any value;
					switch (dataTypes_[columnIndex]) {
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
					case COLUMN_POINT:
						value = PointFactory::FromWkt(field);
						break;					
					case COLUMN_POLYGON:
						value = ComplexPolygonFactory::FromWkt(field);
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
				BOOST_LOG_TRIVIAL(warning) << "Import of table " << tableName_ << " failed on line " << position << " (column " << columnIndex+1 << ")";
				rowData.clear(); 
			}
			catch (std::invalid_argument& e) {
				BOOST_LOG_TRIVIAL(warning) << "Import of file " << tableName_ << " failed on line " << position << " (column " << columnIndex+1 << ")";
				rowData.clear();
			}
		}

		// pushes values of row vector into corresponding columns
		columnIndex = 0;
		for (auto& field : rowData) {
			std::any &wrappedData = data.at(headers_[columnIndex]);
			int v;
			switch (dataTypes_[columnIndex])
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
			case COLUMN_POINT:
				std::any_cast<std::vector<ColmnarDB::Types::Point>&>(wrappedData).push_back(std::any_cast<ColmnarDB::Types::Point>(field));
				break;
			case COLUMN_POLYGON:
				std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>&>(wrappedData).push_back(std::any_cast<ColmnarDB::Types::ComplexPolygon>(field));
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
			for (int i = 0; i < headers_.size(); i++) {
				std::any &wrappedData = data.at(headers_[i]);
				switch (dataTypes_[i])
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
				case COLUMN_POINT:
					std::any_cast<std::vector<ColmnarDB::Types::Point>&>(wrappedData).clear();
					break;
				case COLUMN_POLYGON:
					std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>&>(wrappedData).clear();
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
	inputStream_->clear();
	inputStream_->seekg(0, std::ios::beg);
	aria::csv::CsvParser parser = aria::csv::CsvParser(*inputStream_).delimiter(delimiter_).quote(quotes_);

	int position = 0;
	auto& row = parser.begin();

	int columnIndex = 0;
	for (auto& field : *row) {

		if (position == 0) {
			if (this->header_)
				this->headers_.push_back(field);
			else
				this->headers_.push_back("C" + std::to_string(columnIndex));
		}
			
		columnIndex++;
	}	
}

/// <summary>
/// Extracts types based on 100 leading rows
/// </summary>
void CSVDataImporter::ExtractTypes()
{
	inputStream_->clear();
	inputStream_->seekg(0, std::ios::beg);
	std::vector<std::vector<std::string>> columnData;

	aria::csv::CsvParser parser = aria::csv::CsvParser(*inputStream_).delimiter(delimiter_).quote(quotes_);

	int position = 0;

	for (auto& row : parser) {

		int columnIndex = 0;
		for (auto& field : row) {

			if (position == 0) {
				columnData.push_back(std::vector<std::string>());
			}
			if ((header_ && position > 0) || !header_) {
				columnData[columnIndex].push_back(field);
			}

			columnIndex++;
		}


		position++;

		if (position >= 100)
			break;
	}

	for (auto& column : columnData) {
		DataType type = this->IdentifyDataType(column);
		dataTypes_.push_back(type);
	}
}

/// <summary>
/// Identify data type based on vector of values. Returns maximum type from vector of types.
/// COLUMN_INT < COLUMN_LONG < COLUMN_FLOAT < COLUMN_DOUBLE < COULMN_STRING
/// </summary>
/// <param name="columnValues">vector of string values</param>
/// <returns>Suitable data type</returns>
DataType CSVDataImporter::IdentifyDataType(std::vector<std::string> columnValues)
{
	std::vector<DataType> dataTypes;

	for (auto& s : columnValues) {

		// COLUMN_INT
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

		// COLUMN_LONG
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

		// COLUMN_FLOAT
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

		// COLUMN_DOUBLE
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

		// COLUMN_POINT
		try {
			PointFactory::FromWkt(s);
			dataTypes.push_back(COLUMN_POINT);
			continue;			
		}
		catch (std::invalid_argument& e) {
		}

		// COLUMN_POLYGON
		try {
			ComplexPolygonFactory::FromWkt(s);
			dataTypes.push_back(COLUMN_POLYGON);
			continue;
		}
		catch (std::invalid_argument& e) {
		}

		// COLUMN_STRING
		dataTypes.push_back(COLUMN_STRING);
	}

	if (dataTypes.size() > 0) {
		DataType maxType = dataTypes[0];
		for (auto t : dataTypes) {
			if ((t == COLUMN_POINT && maxType != COLUMN_POINT) || (t != COLUMN_POINT && maxType == COLUMN_POINT)) {
				maxType = COLUMN_STRING;
			}
			else if ((t == COLUMN_POLYGON && maxType != COLUMN_POLYGON) || (t != COLUMN_POLYGON && maxType == COLUMN_POLYGON)) {
				maxType = COLUMN_STRING;
			}
			else if (t > maxType) {
				maxType = t;
			}
		}
		return maxType;
	}
	return COLUMN_STRING;
}

