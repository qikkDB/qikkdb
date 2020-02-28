#pragma once
#include "IColumn.h"
#include "DataType.h"
#include "ConstraintType.h"
#include "Configuration.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <boost/log/trivial.hpp>
#include <boost/asio.hpp>
#ifndef __CUDACC__
#include <any>
#endif

class Database;

/// <summary>
/// The main class representing table containing columns with data.
/// </summary>
class Table
{
private:
    enum CompareResult
    {
        Greater,
        Lower,
        Equal
    };

    const std::shared_ptr<Database>& database;
    std::string name;
    int32_t blockSize_;
    std::unordered_map<std::string, std::unique_ptr<IColumn>> columns;
    std::unordered_map<std::string, std::pair<ConstraintType, std::vector<std::string>>> constraints_;
    std::vector<std::string> sortingColumns;
    std::unique_ptr<std::mutex> columnsMutex_;
    bool saveNecesarry_;
    // save interval in milliseconds, default value is from configuration file, but can be overriden via .db file:
    int32_t saveInterval_ = Configuration::GetInstance().GetDBSaveInterval();
    boost::asio::steady_timer autoSaveDeadline_;

#ifndef __CUDACC__
    void InsertValuesOnSpecificPosition(const std::unordered_map<std::string, std::any>& data,
                                        int indexBlock,
                                        int indexInBlock,
                                        int iterator,
                                        const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks);
    int32_t GetDataRangeInSortingColumn();
    std::tuple<std::vector<std::any>, std::vector<int8_t>>
    GetRowAndBitmaskOfInsertedData(const std::unordered_map<std::string, std::any>& data,
                                   int iterator,
                                   const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks);
    std::tuple<int, int> GetIndicesFromTotalIndex(int index, bool positionToCompare);
    std::tuple<std::vector<std::any>, std::vector<int8_t>> GetRowAndBitmaskOnIndex(int index);
    CompareResult CompareRows(std::vector<std::any> rowToInsert, std::vector<int8_t> maskOfRow, int index);
    std::tuple<int, int> GetIndex(std::vector<std::any> rowToInsert, std::vector<int8_t> maskOfRow);
    int32_t GetDataSizeOfInsertedColumns(const std::unordered_map<std::string, std::any>& data);
    void CheckUniqueConstraintInData(const std::unordered_map<std::string, std::any>& data, int32_t dataSize);
#endif
    void CheckNullableConstraintInData(const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks,
                                       int32_t dataSize);
    bool GetHasUniqueConstraints();
    bool GetHasNotNullConstraints();

public:
    const std::shared_ptr<Database>& GetDatabase();
    const std::string& GetName() const;
    void SetTableName(const std::string& newTableName);
    int32_t GetBlockSize() const;
    void SetBlockSize(int32_t blockSize);
    int32_t GetBlockCount() const;
    int64_t GetSize() const;
    const std::unordered_map<std::string, std::unique_ptr<IColumn>>& GetColumns() const;
    const std::vector<std::string>& GetSortingColumns();
    void SetSortingColumns(std::vector<std::string> columns);
    void AddSortingColumn(const std::string& sortingColumn);
    void RemoveSortingColumn(const std::string& sortingColumn);
    bool GetSaveNecessary() const;
    void SetSaveNecessaryToFalse();
    void RenameColumn(std::string oldColumnName, std::string newColumnName);
    void InsertNullDataIntoNewColumn(std::string newColumnName);
    void AddConstraint(const std::string& constraintName,
                       ConstraintType constraintType,
                       std::vector<std::string> constraintColumns);
    void DropConstraint(const std::string& constraintName);
    const std::unordered_map<std::string, std::pair<ConstraintType, std::vector<std::string>>>&
    GetConstraints() const;
    std::unordered_set<ConstraintType> GetConstraintsForColumn(const std::string& columnName);
    int32_t GetSaveInterval() const;
    void SetSaveInterval(int32_t newSaveInterval);

    /// <summary>
    /// Removes column from columns (in memory).
    /// </summary>
    /// <param name="columnName">Name of column to be removed.</param>
    void EraseColumn(std::string& columnName);

    void AutoSaveTable();

    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.Table"/> class. Also gets from database
    /// the block size and initializes with this value the private variable blockSize. Finally, it initializes columnsMutex_.
    /// </summary>
    /// <param name="database">Pointer to the database which will contains the new table.</param>
    /// <param name="name">Name of the newly created table.</param>
    /// <param name="blockSize">Table block size. If not specified, as the default value a database block size will be used.</param>
    Table(const std::shared_ptr<Database>& database, const char* name, const int32_t blockSize = -1);

    /// <summary>
    /// Insert new column with proper data type into the table.
    /// </summary>
    /// <param name="columnName">Name of column.</param>
    /// <param name="columnType">Data type of column.</param>
    /// <param name="isNullable">Yields if a column can have NULL values. Default value is 'true'.</param>
    /// <param name="isUnique">Yields if a column have unique values. Default value is 'false'.</param>
    void CreateColumn(const char* columnName, DataType columnType, bool isNullable = true, bool isUnique = false);

#ifndef __CUDACC__
    /// <summary>
    /// Insert data into proper column of table considering empty space of last block and maximum size of blocks.
    /// </summary>
    /// <param name="data">Name of column with inserting data.</param>
    /// <param name="compress">Whether data will be compressed.</param>
    void InsertData(const std::unordered_map<std::string, std::any>& data,
                    bool compress = false,
                    const std::unordered_map<std::string, std::vector<int8_t>>& nullMasks =
                        std::unordered_map<std::string, std::vector<int8_t>>());
    int32_t AssignGroupId(std::vector<std::any>& rowData, std::vector<std::unique_ptr<IColumn>>& columns);
#endif

    /// <summary>
    /// Search for column according to its name.
    /// </summary>
    /// <param name="column">Name of column.</param>
    /// <returns>Return true, if table contains particular column. Returns false, if table does not contains particular column.</returns>
    bool ContainsColumn(const char* column);
    std::vector<int32_t> GetTableGroupIds(std::unordered_map<std::string, std::unique_ptr<IColumn>>& columns);
    std::vector<int32_t> GetTableGroupIds(std::vector<std::unique_ptr<IColumn>>& columns);
};
