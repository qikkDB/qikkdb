#include <boost/log/trivial.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <thread>
#include <vector>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>
#include "CSVDataImporter.h"
#include "../CSVParser.hpp"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include "Configuration.h"

/// <summary>
/// Initializes a new instance of the <see cref="T:ColmnarDB.CSVDataImporter"/> class.
/// </summary>
/// <param name="fileName">Path to the CSV file.</param>
/// <param name="header">True, if CSV file has a header. Default value is 'true'.</param>
/// <param name="delimiter">Delimiter between values. Default value is ','.</param>
/// <param name="quotes">Character used for quoting. Default value is '.</param>
/// <param name="decimal">Character used for quoting. Default value is '.</param>
CSVDataImporter::CSVDataImporter(const char* fileName, bool header, char delimiter, char quotes, char decimal)
: inputMapped_(std::make_unique<boost::iostreams::mapped_file>(fileName, boost::iostreams::mapped_file::readonly)),
  tableName_(boost::filesystem::path(fileName).stem().string()), header_(header),
  delimiter_(delimiter), quotes_(quotes), decimal_(decimal)
{
    inputSize_ = inputMapped_.get()->size();
    input_ = inputMapped_.get()->const_data();
    if (std::thread::hardware_concurrency() > 1) numThreads_ = std::thread::hardware_concurrency();
    BOOST_LOG_TRIVIAL(info) << "Import threads: " << numThreads_;
}

/// <summary>
/// Parses CSV file, guess types, creates a single table (if not exists) and fills the table with parsed data.
/// </summary>
/// <param name="database">Database where data will be imported.</param>
/// <param name="sortingColumns">Sorting columns for indices.</param>
/// <param name="tableBlockSize">Block size used for this table.</param>
void CSVDataImporter::ImportTables(std::shared_ptr<Database>& database,
                                   std::vector<std::string> sortingColumns,
                                   int32_t tableBlockSize)
{
    this->ExtractHeaders();
    if (dataTypes_.empty())
    {
        this->ExtractTypes();
    }

    // prepares map columnName -> columnType
    std::unordered_map<std::string, DataType> columns;
    for (int i = 0; i < headers_.size(); i++)
    {
        columns[headers_[i]] = dataTypes_[i];
    }

    // creates table
    Table& table =
        database->CreateTable(columns, tableName_.c_str(), std::unordered_map<std::string, bool>(),
                              std::unordered_map<std::string, bool>(), tableBlockSize);

    if (!sortingColumns.empty())
    {
        table.SetSortingColumns(sortingColumns);
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads_; i++)
    {
        threads.emplace_back(&CSVDataImporter::ParseAndImport, this, i, database->GetBlockSize(),
                             columns, std::ref(table));
    }

    for (int i = 0; i < numThreads_; ++i)
    {
        threads[i].join();
    }
}

/// <summary>
/// Does the parsing and fills the table with parsed data.
/// </summary>
/// <param name="threadId">Id of the thread doing the work.</param>
/// <param name="database">Database where data will be imported.</param>
/// <param name="columns">Columns with names and types: map columnName -> columnType.</param>
/// <param name="table">Table of the database where data will be imported.</param>
void CSVDataImporter::ParseAndImport(int threadId,
                                     int32_t blockSize,
                                     const std::unordered_map<std::string, DataType>& columns,
                                     Table& table)
{

    size_t start = (inputSize_ / numThreads_) * threadId;
    size_t end = start + (inputSize_ / numThreads_);
    if (threadId >= numThreads_ - 1) end = inputSize_;

    // initializes map columnName -> vector of column data
    std::unordered_map<std::string, std::any> data;
    for (int i = 0; i < headers_.size(); i++)
    {
        if (dataTypes_[i] == COLUMN_INT8_T)
        {
            std::vector<int8_t> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_INT)
        {
            std::vector<int32_t> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_LONG)
        {
            std::vector<int64_t> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_FLOAT)
        {
            std::vector<float> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_DOUBLE)
        {
            std::vector<double> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_POINT)
        {
            std::vector<ColmnarDB::Types::Point> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_POLYGON)
        {
            std::vector<ColmnarDB::Types::ComplexPolygon> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else if (dataTypes_[i] == COLUMN_STRING)
        {
            std::vector<std::string> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
        else
        {
            std::vector<std::string> v;
            v.reserve(blockSize);
            data[headers_[i]] = std::move(v);
        }
    }

    aria::csv::CsvParser parser =
        aria::csv::CsvParser(input_, start, end, inputSize_).delimiter(delimiter_).quote(quotes_);

    int position = 0;
    std::vector<std::any> rowData;
    rowData.reserve(headers_.size());

    // parses file and inserts data in batches of size of blockSize
    for (auto& row : parser)
    {

        int columnIndex = 0;
        // skipping first line with header (only relevant in the first thread)
        if (position > 0 || !this->header_ || threadId > 0)
        {
            // casts and puts data into row vector
            // if casting fails, the line is ommited
            try
            {
                for (auto& field : row)
                {
                    std::string fieldCopy = field;
                    boost::algorithm::to_lower(fieldCopy);

                    std::any value;
                    switch (dataTypes_[columnIndex])
                    {
                    case COLUMN_INT8_T:
                        if (fieldCopy == "true")
                        {
                            value = static_cast<int8_t>(1);
                        }
                        else if (fieldCopy == "false")
                        {
                            value = static_cast<int8_t>(0);
                        }
                        break;
                    case COLUMN_INT:
                        value = static_cast<int32_t>(std::stoi(field));
                        break;
                    case COLUMN_LONG:
                        // Check for chars - and : to find out if it is datetime
                        if (boost::contains(field, "-") && boost::contains(field, ":"))
                        {
                            std::tm t;
                            std::istringstream ss(field);
                            ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
                            std::time_t epochTime = std::mktime(&t);
                            value = static_cast<int64_t>(epochTime);
                        }
                        else // If not, parse as raw long long
                        {
                            value = static_cast<int64_t>(std::stoll(field));
                        }
                        break;
                    case COLUMN_FLOAT:
                        value = static_cast<float>(std::stof(field));
                        break;
                    case COLUMN_DOUBLE:
                        value = static_cast<double>(std::stod(field));
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
                    default:
                        throw std::out_of_range("Invalid CSV column data type");
                        break;
                    }
                    rowData.push_back(value);
                    columnIndex++;
                }
            }
            catch (std::out_of_range&)
            {
                BOOST_LOG_TRIVIAL(warning) << "Import of file " << tableName_ << " failed on line "
                                           << position << " (column " << columnIndex + 1 << ")";
                rowData.clear();
            }
            catch (std::invalid_argument&)
            {
                BOOST_LOG_TRIVIAL(warning) << "Import of file " << tableName_ << " failed on line "
                                           << position << " (column " << columnIndex + 1 << ")";
                rowData.clear();
            }
        }

        // pushes values of row vector into corresponding columns
        columnIndex = 0;
        for (auto& field : rowData)
        {
            std::any& wrappedData = data.at(headers_[columnIndex]);
            switch (dataTypes_[columnIndex])
            {
            case COLUMN_INT8_T:
                std::any_cast<std::vector<int8_t>&>(wrappedData).push_back(std::any_cast<int8_t>(field));
                break;
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
                std::any_cast<std::vector<ColmnarDB::Types::Point>&>(wrappedData)
                    .push_back(std::any_cast<ColmnarDB::Types::Point>(field));
                break;
            case COLUMN_POLYGON:
                std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>&>(wrappedData)
                    .push_back(std::any_cast<ColmnarDB::Types::ComplexPolygon>(field));
                break;
            case COLUMN_STRING:
                std::any_cast<std::vector<std::string>&>(wrappedData).push_back(std::any_cast<std::string>(field));
                break;
            default:
                throw std::out_of_range("Invalid CSV column data type");
                break;
            }

            columnIndex++;
        }

        rowData.clear();

        position++;

        // inserts parsed data into database when blockSize reached
        if (position % blockSize == 0)
        {

            insertMutex_.lock();
            table.InsertData(data, Configuration::GetInstance().IsUsingCompression());
            insertMutex_.unlock();

            // clears parsed data so far
            for (int i = 0; i < headers_.size(); i++)
            {
                std::any& wrappedData = data.at(headers_[i]);
                switch (dataTypes_[i])
                {
                case COLUMN_INT8_T:
                    std::any_cast<std::vector<int8_t>&>(wrappedData).clear();
                    break;
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
                default:
                    throw std::out_of_range("Invalid CSV column data type");
                    break;
                }
            }
        }
    }

    // inserts remaing rows into table
    insertMutex_.lock();
    table.InsertData(data, Configuration::GetInstance().IsUsingCompression());
    insertMutex_.unlock();
}


/// <summary>
/// Extracts column names from header. If there is no header, column names are created C0, C1, ... and so on.
/// </summary>
void CSVDataImporter::ExtractHeaders()
{
    aria::csv::CsvParser parser =
        aria::csv::CsvParser(input_, 0, inputSize_, inputSize_).delimiter(delimiter_).quote(quotes_);

    int position = 0;
    auto row = parser.begin();

    int columnIndex = 0;
    for (auto& field : *row)
    {

        if (position == 0)
        {
            if (this->header_)
                this->headers_.push_back(field);
            else
                this->headers_.push_back("C" + std::to_string(columnIndex));
        }

        columnIndex++;
    }
}

/// <summary>
/// Extracts types based on 100 leading rows.
/// </summary>
void CSVDataImporter::ExtractTypes()
{
    std::vector<std::vector<std::string>> columnData;

    aria::csv::CsvParser parser =
        aria::csv::CsvParser(input_, 0, inputSize_, inputSize_).delimiter(delimiter_).quote(quotes_);

    int position = 0;

    for (auto& row : parser)
    {

        int columnIndex = 0;
        for (auto& field : row)
        {

            if (position == 0)
            {
                columnData.push_back(std::vector<std::string>());
            }
            if ((header_ && position > 0) || !header_)
            {
                columnData[columnIndex].push_back(field);
            }

            columnIndex++;
        }


        position++;

        if (position >= 100) break;
    }

    for (auto& column : columnData)
    {
        DataType type = this->IdentifyDataType(column);
        dataTypes_.push_back(type);
    }
}

/// <summary>
/// Identify data type based on vector of values. Returns maximum type from vector of types.
/// COLUMN_INT < COLUMN_LONG < COLUMN_FLOAT < COLUMN_DOUBLE < COULMN_STRING.
/// </summary>
/// <param name="columnValues">Vector of string values.</param>
/// <returns>Suitable data type.</returns>
DataType CSVDataImporter::IdentifyDataType(std::vector<std::string> columnValues)
{
    std::vector<DataType> dataTypes;

    for (auto& s : columnValues)
    {
        // COLUMN_INT8_T
        std::string sCopy = s;
        boost::algorithm::to_lower(sCopy);

        if (sCopy == "true")
        {
            dataTypes.push_back(COLUMN_INT8_T);
            continue;
        }
        else if (sCopy == "false")
        {
            dataTypes.push_back(COLUMN_INT8_T);
            continue;
        }

        // COLUMN_INT
        try
        {
            size_t position;
            std::stoi(s, &position);
            if (s.length() == position)
            {
                dataTypes.push_back(COLUMN_INT);
                continue;
            }
        }
        catch (std::out_of_range&)
        {
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_LONG
        try
        {
            size_t position;
            std::stoll(s, &position);
            if (s.length() == position)
            {
                dataTypes.push_back(COLUMN_LONG);
                continue;
            }
        }
        catch (std::out_of_range&)
        {
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_FLOAT
        try
        {
            size_t position;
            std::stof(s, &position);
            if (s.length() == position)
            {
                dataTypes.push_back(COLUMN_FLOAT);
                continue;
            }
        }
        catch (std::out_of_range&)
        {
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_DOUBLE
        try
        {
            size_t position;
            std::stod(s, &position);
            if (s.length() == position)
            {
                dataTypes.push_back(COLUMN_DOUBLE);
                continue;
            }
        }
        catch (std::out_of_range&)
        {
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_POINT
        try
        {
            PointFactory::FromWkt(s);
            dataTypes.push_back(COLUMN_POINT);
            continue;
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_POLYGON
        try
        {
            ComplexPolygonFactory::FromWkt(s);
            dataTypes.push_back(COLUMN_POLYGON);
            continue;
        }
        catch (std::invalid_argument&)
        {
        }

        // COLUMN_STRING
        dataTypes.push_back(COLUMN_STRING);
    }

    if (dataTypes.size() > 0)
    {
        DataType maxType = dataTypes[0];
        for (auto t : dataTypes)
        {
            if ((t == COLUMN_POINT && maxType != COLUMN_POINT) || (t != COLUMN_POINT && maxType == COLUMN_POINT))
            {
                maxType = COLUMN_STRING;
            }
            else if ((t == COLUMN_POLYGON && maxType != COLUMN_POLYGON) ||
                     (t != COLUMN_POLYGON && maxType == COLUMN_POLYGON))
            {
                maxType = COLUMN_STRING;
            }
            else if (t > maxType)
            {
                maxType = t;
            }
        }
        return maxType;
    }
    return COLUMN_STRING;
}

/// <summary>
/// Sets column types based on pre determined values.
/// Disables type guessing.
/// </summary>
/// <param name="columnValues">Vector of types values.</param>
void CSVDataImporter::SetTypes(const std::vector<DataType>& types)
{
    dataTypes_ = types;
}

/// <summary>
/// Sets table name.
/// Disables table name guessing according to filename.
/// </summary>
/// <param name="tableName">Table name.</param>
void CSVDataImporter::SetTableName(const std::string tableName)
{
    tableName_ = tableName;
}