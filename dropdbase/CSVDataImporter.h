#pragma once
#include <boost/iostreams/device/mapped_file.hpp>
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
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.CSVDataImporter"/> class. Imports data from string.
	/// </summary>
	/// <param name="inputString">Input string of data which will be imported.</param>
	/// <param name="tableName">Name of table into which data will be imported.</param>
	/// <param name="header">True, if CSV file has a header. Default value is 'true'.</param>
	/// <param name="delimiter">Delimiter between values. Default value is ','.</param>
	/// <param name="quotes">Character used for quoting. Default value is '.</param>
	/// <param name="decimal">Character used as decimal point. Default value is '.'.</param>
	CSVDataImporter(const char* inputString, const char* tableName, bool header = true, char delimiter = ',', char quotes = '\'', char decimal = '.') :
		input_(inputString),
		inputSize_(strlen(inputString)),
		tableName_(tableName),
		header_(header),
		delimiter_(delimiter),
		quotes_(quotes),
		decimal_(decimal)
	{			
	}
	
	/// <summary>
	/// Initializes a new instance of the <see cref="T:ColmnarDB.CSVDataImporter"/> class. Imports data from CSV file.
	/// </summary>
	/// <param name="fileName">Path to the CSV file.</param>
	/// <param name="header">True, if CSV file has a header. Default value is 'true'.</param>
	/// <param name="delimiter">Delimiter between values. Default value is ','.</param>
	/// <param name="quotes">Character used for quoting. Default value is '.</param>
	/// <param name="decimal">Character used as decimal point. Default value is '.'.</param>
	CSVDataImporter(const char* fileName, bool header = true, char delimiter = ',', char quotes = '\'', char decimal = '.');
	void ImportTables(std::shared_ptr<Database>& database, std::vector<std::string> sortingColumns = {});
	void ExtractHeaders();
	void ExtractTypes();
	void SetTypes(const std::vector<DataType>& types);
	void SetTableName(const std::string tableName);
	int GetNumberOfThreads() const { return numThreads_; }
	void SetNumberOfThreads(const int numThreads) { numThreads_ = numThreads; }
private:
	std::unique_ptr<std::istream> inputStream_;
	std::unique_ptr<boost::iostreams::mapped_file> inputMapped_;
	const char* input_;
	size_t inputSize_;
	std::string tableName_;
	bool header_;
	char delimiter_;
	char quotes_;
	char decimal_;
	std::vector<std::string> headers_;
	std::vector<DataType> dataTypes_;
	int numThreads_ = 1;
	std::mutex insertMutex_;

	DataType IdentifyDataType(std::vector<std::string> columnValues);
	void ParseAndImport(int threadId, int32_t blockSize, const std::unordered_map<std::string, DataType>& columns, Table& table);
};
