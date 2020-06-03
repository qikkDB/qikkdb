#pragma once
#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include <limits>
#include <boost/log/trivial.hpp>

#include "DataType.h"
#include "ConstraintType.h"
#include "QueryEngine/Context.h"
#include "ColumnBase.h"
#include "Table.h"
#include "BlockBase.h"
#include "Types/ComplexPolygon.pb.h"

/// <summary>
/// The main class representing database containing tables with data.
/// </summary>

class IColumn;

class Database
{
    friend class DatabaseGenerator;

private:
    static std::mutex dbAccessMutex_;
    static std::mutex dbFilesMutex_;
    std::string name_;
    int32_t blockSize_;
    std::unordered_map<std::string, Table> tables_;

    /// <summary>
    /// Load column of a table into memory from disk.
    /// </summary>
    /// <param name="fileDbPath">Path to DB_EXTENSION file.</param>
    /// <param name="fileAddressPath">Path to COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="fileDataPath">Path to COLUMN_DATA_EXTENSION file.</param>
    /// <param name="fileFragmentPath">Path to FRAGMENT_DATA_EXTENSION file.</param>
    /// <param name="encoding">Encoding of the string data in FRAGMENT_DATA_EXTENSION file.</param>
    /// <param name="persistenceFormatVersion">Version of format used to persist DB_EXTENSION and
    /// COLUMN_DATA_EXTENSION files into disk.</param> <param name="type">Type of column according
    /// to DataType enumeration.</param>
    /// <param name="isNullable">Flag if a column can have NULL values.</param>
    /// <param name="isUnique">Flag if a column can have only unique values and not
    /// a single one NULL value.</param>
    /// <param name="defaultValue">Default column value in string format.</param>
    ///< param name="table">Instance of table into which the column
    /// should be added.</param> <param name="columnName">Names of particular column.</param>
    static void LoadColumn(const std::string fileDbPath,
                           const std::string fileAddressPath,
                           const std::string fileDataPath,
                           const std::string fileFragmentPath,
                           const std::string encoding,
                           const int32_t persistenceFormatVersion,
                           const int32_t type,
                           const bool isNullable,
                           const bool isUnique,
                           const std::string defaultValue,
                           Table& table,
                           const std::string columnName);

    /// <summary>
    /// Write single ComplexPolygon data into disk. It has to seek the block's position
    /// in the COLUMN_DATA_EXTENSION file and replace the block's data with the data wich is in memory.
    /// </summary>
    /// <param name="table">Name of the particular table.</param>
    /// <param name="column">Name of the column to which the block belongs to.</param>
    /// <param name="block">Block wich is going to be persisted.</param>
    /// <param name="fragmentPosition">Block position saved in COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="dataPosition">Block position of COLUMN_DATA_EXTENSION file, used only in ComplexPolygon
    /// and String block types.</param>
    /// <param name="dbName">Name of the database.</param>
    static void WriteBlockPolygonType(const Table& table,
                                      const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                                      BlockBase<ColmnarDB::Types::ComplexPolygon>& block,
                                      const uint64_t fragmentPosition,
                                      const uint64_t dataPosition,
                                      const std::string dbName)
    {
        const int32_t blockSize = table.GetBlockSize();
        std::string fileDataPath = column.second->GetFileDataPath();
        const std::string tableName = table.GetName();

        // default data path if not specified by user:
        if (fileDataPath.size() <= Configuration::GetInstance().GetDatabaseDir().size())
        {
            fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName + SEPARATOR +
                           tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
        }

        std::ofstream colDataFile(fileDataPath, std::ios::binary);

        if (colDataFile.is_open())
        {
            const int32_t type = column.second->GetColumnType();
            const bool isNullable = column.second->GetIsNullable();
            const bool isUnique = column.second->GetIsUnique();

            uint32_t index = block.GetIndex();

            const ColumnBase<ColmnarDB::Types::ComplexPolygon>& colPolygon =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>&>(*(column.second));

            std::string fileFragmentPath = colPolygon.GetFileFragmentPath();

            // default data path if not specified by user:
            if (fileFragmentPath.size() <= Configuration::GetInstance().GetDatabaseDir().size())
            {
                fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName +
                                   SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                   FRAGMENT_DATA_EXTENSION;
            }

            std::ofstream colFragDataFile(fileFragmentPath, std::ios::binary);

            // persist block data into disk:
            colFragDataFile.seekp(fragmentPosition);

            BOOST_LOG_TRIVIAL(debug)
                << "Database: Saving block of ComplexPolygon data with index = " << index;

            auto data = block.GetData();
            int32_t groupId = block.GetGroupId();
            size_t blockCurrentSize = block.GetSize();
            int64_t dataByteSize = 0;

            colDataFile.seekp(dataPosition);
            colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write block index
            colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write group id (binary index)

            if (isNullable)
            {
                int32_t nullBitMaskLength = (block.GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                  sizeof(int32_t)); // write nullBitMask length
                colDataFile.write(reinterpret_cast<char*>(block.GetNullBitmask()),
                                  nullBitMaskLength); // write nullBitMask
            }

            colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize), sizeof(uint64_t)); // write number of entries

            if (colFragDataFile.is_open())
            {

                // write string data (entries in WKT format) into polygon fragment data file:
                for (int32_t i = 0; i < blockCurrentSize; i++)
                {
                    // transform protobuf message into WKT strings:
                    std::string wktPolygon = ComplexPolygonFactory::WktFromPolygon(data[i]);

                    // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                    dataByteSize += wktPolygon.length() + 1 + sizeof(int32_t);

                    if (dataByteSize <= FRAGMENT_SIZE_BYTES)
                    {
                        // writing entries that fit into a fragment
                        int32_t entryByteLength = wktPolygon.length() + 1; // +1 because '\0'

                        colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // write entry length
                        colFragDataFile.write(wktPolygon.c_str(), entryByteLength); // write entry data
                    }
                    else
                    {
                        // there is still some data which will be saved into next fragment:
                        // padding the not full fragment to it's maximum size, so the size of fragment is always fixed:
                        if (dataByteSize - (wktPolygon.length() + 1 + sizeof(int32_t)) < FRAGMENT_SIZE_BYTES)
                        {
                            const int32_t freeSpaceByteLength =
                                FRAGMENT_SIZE_BYTES -
                                (dataByteSize - (wktPolygon.length() + 1 + sizeof(int32_t)));

                            // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                            if (freeSpaceByteLength >= sizeof(int32_t))
                            {
                                int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                                std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(emptyStringData.c_str(),
                                                      freeSpaceByteLength - sizeof(int32_t)); // write empty data
                            }
                            else
                            {
                                // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                // do not write int32_t header, because there is no space for it
                                std::string emptyStringData(freeSpaceByteLength, '*');
                                colFragDataFile.write(emptyStringData.c_str(),
                                                      freeSpaceByteLength); // write empty data
                            }
                        }

                        // write the actual entry into another fragment (create a new fragment) and change data byte size:
                        // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                        dataByteSize = wktPolygon.length() + 1 + sizeof(int32_t);

                        // writing entries that fit into a fragment
                        int32_t entryByteLength = wktPolygon.length() + 1; // +1 because '\0'
                        colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // write entry length
                        colFragDataFile.write(wktPolygon.c_str(), entryByteLength); // write entry data
                    }

                    // padding the not full fragment, when there is no data to be saved in another fragmet
                    if (i == blockCurrentSize - 1)
                    {
                        const int32_t freeSpaceByteLength = FRAGMENT_SIZE_BYTES - dataByteSize;

                        // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                        if (freeSpaceByteLength >= sizeof(int32_t))
                        {
                            int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                            std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                            colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                  sizeof(int32_t)); // write entry length
                            colFragDataFile.write(emptyStringData.c_str(),
                                                  freeSpaceByteLength - sizeof(int32_t)); // write empty data
                        }
                        else
                        {
                            // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                            // do not write int32_t header, because there is no space for it
                            std::string emptyStringData(freeSpaceByteLength, '*');
                            colFragDataFile.write(emptyStringData.c_str(),
                                                  freeSpaceByteLength); // write empty data
                        }
                    }
                }
            }
            else
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: WriteBlockPolygonType, case1 - Could not open file " +
                           std::string(Configuration::GetInstance().GetDatabaseDir() + dbName +
                                       SEPARATOR + tableName + SEPARATOR +
                                       column.second->GetName() + FRAGMENT_DATA_EXTENSION) +
                           " for writing. Persisting "
                    << FRAGMENT_DATA_EXTENSION
                    << " file was not successful. Check if the process "
                       "have write access into the folder or file.";
            }

            /* check if we did not get UINT32_MAX value in index - this value is reserved
            to identify new block, which are just in memory and have never been persisted
            into disk. If index reached this value, it means, the blockSize had been chosen
            to too small value and we have reached our maximum number of blocks. No new
            blocks will be persisted in order to at least save the current data.*/
            if (index == UINT32_MAX)
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: When saving block of data into file: " << fileDataPath
                    << " tha maximum number of block has been reached. For that "
                       "reason, this block of data and data of other blocks whose have "
                       "not "
                       "been persisted yet, will not be persisted in order to protect "
                       "already persisted data on disk.";
            }

            colFragDataFile.close();
        }
        else
        {
            BOOST_LOG_TRIVIAL(error)
                << "ERROR: Database: WriteBlockPolygonType, case2 - Could not open file " +
                       std::string(Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR + tableName +
                                   SEPARATOR + column.second->GetName() + FRAGMENT_DATA_EXTENSION) +
                       " for writing. Persisting "
                << FRAGMENT_DATA_EXTENSION
                << " file was not successful. Check if the process "
                   "have write access into the folder or file.";
        }

        /* check if we did not get UINT32_MAX value in index - this value is reserved
        to identify new block, which are just in memory and have never been persisted
        into disk. If index reached this value, it means, the blockSize had been chosen
        to too small value and we have reached our maximum number of blocks. No new
        blocks will be persisted in order to at least save the current data.*/
        if (block.GetIndex() == UINT32_MAX)
        {
            BOOST_LOG_TRIVIAL(error)
                << "ERROR: Database: When saving block of data into file: " << fileDataPath
                << " tha maximum number of block has been reached. For that "
                   "reason, this block of data and data of other blocks whose have "
                   "not "
                   "been persisted yet, will not be persisted in order to protect "
                   "already persisted data on disk.";
        }
    }

    /// <summary>
    /// Write single string data into disk. It has to seek the block's position
    /// in the COLUMN_DATA_EXTENSION file and replace the block's data with the data wich is in memory.
    /// </summary>
    /// <param name="table">Name of the particular table.</param>
    /// <param name="column">Name of the column to which the block belongs to.</param>
    /// <param name="block">Block wich is going to be persisted.</param>
    /// <param name="fragmentPosition">Block position saved in COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="dataPosition">Block position of COLUMN_DATA_EXTENSION file, used only in
    /// ComplexPolygon and String block types.</param>
    /// <param name="dbName">Name of the database.</param>
    static void WriteBlockStringType(const Table& table,
                                     const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                                     BlockBase<std::string>& block,
                                     const uint64_t fragmentPosition,
                                     const uint64_t dataPosition,
                                     const std::string dbName)
    {
        const int32_t blockSize = table.GetBlockSize();
        std::string fileDataPath = column.second->GetFileDataPath();
        const std::string tableName = table.GetName();

        // default data path if not specified by user:
        if (fileDataPath.size() <= Configuration::GetInstance().GetDatabaseDir().size())
        {
            fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName + SEPARATOR +
                           tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
        }

        std::ofstream colDataFile(fileDataPath, std::ios::binary);

        if (colDataFile.is_open())
        {
            const int32_t type = column.second->GetColumnType();
            const bool isNullable = column.second->GetIsNullable();
            const bool isUnique = column.second->GetIsUnique();

            uint32_t index = block.GetIndex();

            const ColumnBase<std::string>& colStr =
                dynamic_cast<const ColumnBase<std::string>&>(*(column.second));

            std::string fileFragmentPath = colStr.GetFileFragmentPath();

            // default data path if not specified by user:
            if (fileFragmentPath.size() <= Configuration::GetInstance().GetDatabaseDir().size())
            {
                fileFragmentPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName +
                                   SEPARATOR + tableName + SEPARATOR + column.second->GetName() +
                                   FRAGMENT_DATA_EXTENSION;
            }

            std::ofstream colFragDataFile(fileFragmentPath, std::ios::binary);

            // persist block data into disk:
            colFragDataFile.seekp(fragmentPosition);

            BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of String data with index = " << index;

            auto data = block.GetData();
            int32_t groupId = block.GetGroupId();
            size_t blockCurrentSize = block.GetSize();
            int64_t dataByteSize = 0;

            colDataFile.seekp(dataPosition);
            colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
            colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId

            if (isNullable)
            {
                int32_t nullBitMaskLength = (block.GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                  sizeof(int32_t)); // write nullBitMask length
                colDataFile.write(reinterpret_cast<char*>(block.GetNullBitmask()),
                                  nullBitMaskLength); // write nullBitMask
            }

            colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                              sizeof(uint64_t)); // write block length (number of entries)


            if (colFragDataFile.is_open())
            {
                // write string data (entries) into string data file:
                for (int32_t i = 0; i < blockCurrentSize; i++)
                {
                    // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                    dataByteSize += data[i].length() + 1 + sizeof(int32_t);

                    if (dataByteSize <= FRAGMENT_SIZE_BYTES)
                    {
                        // writing entries that fit into a fragment
                        int32_t entryByteLength = data[i].length() + 1; // +1 because '\0'

                        colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // write entry length
                        colFragDataFile.write(data[i].c_str(), entryByteLength); // write entry data
                    }
                    else
                    {
                        // there is still some data which will be saved into next fragment:
                        // padding the not full fragment to it's maximum size, so the size of fragment is always fixed:
                        if (dataByteSize - (data[i].length() + 1 + sizeof(int32_t)) < FRAGMENT_SIZE_BYTES)
                        {
                            const int32_t freeSpaceByteLength =
                                FRAGMENT_SIZE_BYTES -
                                (dataByteSize - (data[i].length() + 1 + sizeof(int32_t)));

                            // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                            if (freeSpaceByteLength >= sizeof(int32_t))
                            {
                                int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                                std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                                colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                      sizeof(int32_t)); // write entry length
                                colFragDataFile.write(emptyStringData.c_str(),
                                                      freeSpaceByteLength - sizeof(int32_t)); // write empty data
                            }
                            else
                            {
                                // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                                // do not write int32_t header, because there is no space for it
                                std::string emptyStringData(freeSpaceByteLength, '*');
                                colFragDataFile.write(emptyStringData.c_str(),
                                                      freeSpaceByteLength); // write empty data
                            }
                        }

                        // write the actual entry into another fragment (create a new fragment) and change data byte size:
                        // +1 because '\0', +sizeof(int32_t) because each string is prefixed it's length
                        dataByteSize = data[i].length() + 1 + sizeof(int32_t);

                        // writing entries that fit into a fragment
                        int32_t entryByteLength = data[i].length() + 1; // +1 because '\0'
                        colFragDataFile.write(reinterpret_cast<char*>(&entryByteLength),
                                              sizeof(int32_t)); // write entry length
                        colFragDataFile.write(data[i].c_str(), entryByteLength); // write entry data
                    }

                    // padding the not full fragment, when there is no data to be saved in another fragmet
                    if (i == blockCurrentSize - 1)
                    {
                        const int32_t freeSpaceByteLength = FRAGMENT_SIZE_BYTES - dataByteSize;

                        // if there is enough space for padding with int32_t header which tells us how many bytes are padded:
                        if (freeSpaceByteLength >= sizeof(int32_t))
                        {
                            int32_t freeSpaceByteLengthNeg = -(freeSpaceByteLength - sizeof(int32_t));
                            std::string emptyStringData(freeSpaceByteLength - sizeof(int32_t), '*');

                            colFragDataFile.write(reinterpret_cast<char*>(&freeSpaceByteLengthNeg),
                                                  sizeof(int32_t)); // write entry length
                            colFragDataFile.write(emptyStringData.c_str(),
                                                  freeSpaceByteLength - sizeof(int32_t)); // write empty data
                        }
                        else
                        {
                            // just write as many '*' as there are free bytes to ensure fixed fragment byte size
                            // do not write int32_t header, because there is no space for it
                            std::string emptyStringData(freeSpaceByteLength, '*');
                            colFragDataFile.write(emptyStringData.c_str(),
                                                  freeSpaceByteLength); // write empty data
                        }
                    }
                }
            }
            else
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: WriteBlockStringType - Could not open file " +
                           std::string(Configuration::GetInstance().GetDatabaseDir() + dbName +
                                       SEPARATOR + tableName + SEPARATOR +
                                       column.second->GetName() + FRAGMENT_DATA_EXTENSION) +
                           " for writing. Persisting "
                    << FRAGMENT_DATA_EXTENSION
                    << " file was not successful. Check if the process "
                       "have write access into the folder or file.";
            }

            /* check if we did not get UINT32_MAX value in index - this value is reserved
            to identify new block, which are just in memory and have never been persisted
            into disk. If index reached this value, it means, the blockSize had been chosen
            to too small value and we have reached our maximum number of blocks. No new
            blocks will be persisted in order to at least save the current data.*/
            if (block.GetIndex() == UINT32_MAX)
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: When saving block of data into file: " << fileDataPath
                    << " tha maximum number of block has been reached. For that "
                       "reason, this block of data and data of other blocks whose have "
                       "not "
                       "been persisted yet, will not be persisted in order to protect "
                       "already persisted data on disk.";
            }

            colFragDataFile.close();
        }
    }

    /// <summary>
    /// Write single block of int8_t, int32_t, int64_t, float, double data into disk. It has to seek the block's position
    /// in the COLUMN_DATA_EXTENSION file and replace the block's data with the data wich is in memory.
    /// </summary>
    /// <param name="table">Name of the particular table.</param>
    /// <param name="column">Name of the column to which the block belongs to.</param>
    /// <param name="block">Block wich is going to be persisted.</param>
    /// <param name="blockPosition">Block position saved in COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="dbName">Name of the database.</param>
    /// and String block types.</param>
    template <typename T>
    static void WriteBlockNumericTypes(const Table& table,
                                       const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                                       BlockBase<T>& block,
                                       const uint64_t blockPosition,
                                       const std::string dbName)
    {
        const int32_t blockSize = table.GetBlockSize();
        std::string fileDataPath = column.second->GetFileDataPath();
        const std::string tableName = table.GetName();

        // default data path if not specified by user:
        if (fileDataPath.size() == 0 || fileDataPath == Configuration::GetInstance().GetDatabaseDir())
        {
            fileDataPath = Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR +
                           tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
        }

        std::ofstream colDataFile(fileDataPath, std::ios::binary);

        if (colDataFile.is_open())
        {
            const int32_t type = column.second->GetColumnType();
            const bool isNullable = column.second->GetIsNullable();
            const bool isUnique = column.second->GetIsUnique();


            uint32_t index = block.GetIndex();

            const ColumnBase<T>& colInt = dynamic_cast<const ColumnBase<T>&>(*(column.second));

            // persist block data into disk:
            colDataFile.seekp(blockPosition);

            BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Int8 data with index = " << index;

            auto data = block.GetData();
            size_t blockCurrentSize = block.GetSize();
            std::unique_ptr<T[]> emptyData(new T[blockSize - blockCurrentSize]);
            std::fill(emptyData.get(), emptyData.get() + (blockSize - blockCurrentSize),
                      std::numeric_limits<T>::max());
            bool isCompressed = block.IsCompressed();
            int32_t groupId = block.GetGroupId();
            T min = block.GetMin();
            T max = block.GetMax();
            float avg = block.GetAvg();
            T sum = block.GetSum();

            colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
            colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId
            if (isNullable)
            {
                int32_t nullBitMaskLength = (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                  sizeof(int32_t)); // write nullBitMask length
                colDataFile.write(reinterpret_cast<char*>(block.GetNullBitmask()),
                                  nullBitMaskLength); // write nullBitMask
            }
            colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                              sizeof(uint64_t)); // write block length (number of entries)
            colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed
            colDataFile.write(reinterpret_cast<char*>(&min), sizeof(T)); // write statistics min
            colDataFile.write(reinterpret_cast<char*>(&max), sizeof(T)); // write statistics max
            colDataFile.write(reinterpret_cast<char*>(&avg), sizeof(float)); // write statistics avg
            colDataFile.write(reinterpret_cast<char*>(&sum), sizeof(T)); // write statistics sum
            colDataFile.write(reinterpret_cast<const char*>(data),
                              blockCurrentSize * sizeof(T)); // write block of data
            colDataFile.write(reinterpret_cast<const char*>(emptyData.get()),
                              (blockSize - blockCurrentSize) * sizeof(T)); // write empty entries as well

            int32_t nullBitMaskLength = (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

            /* check if we did not get UINT32_MAX value in index - this value is reserved
            to identify new block, which are just in memory and have never been persisted
            into disk. If index reached this value, it means, the blockSize had been chosen
            to too small value and we have reached our maximum number of blocks. No new
            blocks will be persisted in order to at least save the current data.*/
            if (index == UINT32_MAX)
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: When saving block of data into file: " << fileDataPath
                    << " tha maximum number of block has been reached. For that "
                       "reason, this block of data and data of other blocks whose have "
                       "not "
                       "been persisted yet, will not be persisted in order to protect "
                       "already persisted data on disk.";
            }

            colDataFile.close();
        }
        else
        {
            BOOST_LOG_TRIVIAL(error)
                << "ERROR: Database: WriteBlockNumericTypes - Could not open file " +
                       std::string(Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR +
                                   tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION) +
                       " for writing. Persisting "
                << COLUMN_DATA_EXTENSION
                << " file was not successful. Check if the process "
                   "have write access into the folder or file.";
        }
    }

    /// <summary>
    /// Write single block Point data into disk. It has to seek the block's position
    /// in the COLUMN_DATA_EXTENSION file and replace the block's data with the data wich is in memory.
    /// </summary>
    /// <param name="table">Name of the particular table.</param>
    /// <param name="column">Name of the column to which the block belongs to.</param>
    /// <param name="block">Block wich is going to be persisted.</param>
    /// <param name="blockPosition">Block position saved in COLUMN_ADDRESS_EXTENSION file.</param>
    /// <param name="dbName">Name of the database.</param>
    /// and String block types.</param>
    template <>
    static void
    WriteBlockNumericTypes<ColmnarDB::Types::Point>(const Table& table,
                                                    const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                                                    BlockBase<ColmnarDB::Types::Point>& block,
                                                    const uint64_t blockPosition,
                                                    const std::string dbName)
    {
        int32_t blockSize = table.GetBlockSize();
        std::string fileDataPath = column.second->GetFileDataPath();
        const std::string tableName = table.GetName();

        // default data path if not specified by user:
        if (fileDataPath.size() <= Configuration::GetInstance().GetDatabaseDir().size())
        {
            fileDataPath = Configuration::GetInstance().GetDatabaseDir().c_str() + dbName + SEPARATOR +
                           tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION;
        }

        std::ofstream colDataFile(fileDataPath, std::ios::binary);

        if (colDataFile.is_open())
        {
            const int32_t type = column.second->GetColumnType();
            const bool isNullable = column.second->GetIsNullable();
            const bool isUnique = column.second->GetIsUnique();

            uint32_t index = block.GetIndex();

            const ColumnBase<ColmnarDB::Types::Point>& colPoint =
                dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>&>(*(column.second));

            // persist block data into disk:
            colDataFile.seekp(blockPosition);

            BOOST_LOG_TRIVIAL(debug) << "Database: Saving block of Point data with index = " << index;

            auto data = block.GetData();
            int32_t groupId = block.GetGroupId();
            size_t blockCurrentSize = block.GetSize();
            bool isCompressed = block.IsCompressed();

            colDataFile.write(reinterpret_cast<char*>(&index), sizeof(uint32_t)); // write index
            colDataFile.write(reinterpret_cast<char*>(&groupId), sizeof(int32_t)); // write groupId

            if (isNullable)
            {
                int32_t nullBitMaskLength = (block.GetSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                colDataFile.write(reinterpret_cast<char*>(&nullBitMaskLength),
                                  sizeof(int32_t)); // write nullBitMask length
                colDataFile.write(reinterpret_cast<char*>(block.GetNullBitmask()),
                                  nullBitMaskLength); // write nullBitMask
            }

            colDataFile.write(reinterpret_cast<char*>(&blockCurrentSize),
                              sizeof(uint64_t)); // write block length (number of entries)
            colDataFile.write(reinterpret_cast<char*>(&isCompressed), sizeof(bool)); // write whether compressed

            // write entries:
            for (size_t i = 0; i < blockCurrentSize; i++)
            {
                float latitude = data[i].geopoint().latitude();
                float longitude = data[i].geopoint().longitude();

                colDataFile.write(reinterpret_cast<char*>(&latitude), sizeof(float)); // write latitude
                colDataFile.write(reinterpret_cast<char*>(&longitude), sizeof(float)); // write longitude
            }

            // padding to block size with std::numeric_limits<float>::max() values:
            for (size_t i = blockCurrentSize; i < blockSize; i++)
            {
                float value = std::numeric_limits<float>::max();

                colDataFile.write(reinterpret_cast<char*>(&value), sizeof(float)); // write latitude
                colDataFile.write(reinterpret_cast<char*>(&value), sizeof(float)); // write longitude
            }

            const int32_t nullBitMaskLength = (blockCurrentSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

            /* check if we did not get UINT32_MAX value in index - this value is reserved
            to identify new block, which are just in memory and have never been persisted
            into disk. If index reached this value, it means, the blockSize had been chosen
            to too small value and we have reached our maximum number of blocks. No new
            blocks will be persisted in order to at least save the current data.*/
            if (index == UINT32_MAX)
            {
                BOOST_LOG_TRIVIAL(error)
                    << "ERROR: Database: When saving block of data into file: " << fileDataPath
                    << " tha maximum number of block has been reached. For that "
                       "reason, this block of data and data of other blocks whose have "
                       "not "
                       "been persisted yet, will not be persisted in order to protect "
                       "already persisted data on disk.";
            }

            colDataFile.close();
        }
        else
        {
            BOOST_LOG_TRIVIAL(error)
                << "ERROR: Database: WriteBlockNumericTypes<ColmnarDB::Types::Point> - Could not "
                   "open file " +
                       std::string(Configuration::GetInstance().GetDatabaseDir() + dbName + SEPARATOR +
                                   tableName + SEPARATOR + column.second->GetName() + COLUMN_DATA_EXTENSION) +
                       " for writing. Persisting "
                << COLUMN_DATA_EXTENSION
                << " file was not successful. Check if the process "
                   "have write access into the folder or file.";
        }
    }

    /// <summary>
    /// Write column into disk (all it's blocks).
    /// </summary>
    /// <param name="column">Column to be written.</param>
    /// <param name="dbName">Name of the database.</param>
    /// <param name="table">Name of the particular table.</param>
    static void WriteColumn(const std::pair<const std::string, std::unique_ptr<IColumn>>& column,
                            const std::string dbName,
                            const Table& table);

public:
    static constexpr const int32_t PERSISTENCE_FORMAT_VERSION = 1;
    static constexpr const char* SEPARATOR = "@";
    static constexpr const char* DB_EXTENSION = ".db";
    static constexpr const char* COLUMN_DATA_EXTENSION = ".data";
    static constexpr const char* COLUMN_ADDRESS_EXTENSION = ".adrs";
    static constexpr const char* FRAGMENT_DATA_EXTENSION = ".fragdata";
    static constexpr const int32_t FRAGMENT_SIZE_BYTES = 1048576;
    static constexpr const char* POLYGON_DEFAULT_VALUE = "POLYGON((0 0, 1 1, 2 2, 0 0))";
    static std::mutex dbMutex_;
    /// <summary>
    /// Initializes a new instance of the <see cref="T:ColmnarDB.Database"/> class.
    /// </summary>
    /// <param name="databaseName">Database name.</param>
    /// <param name="blockSize">Block size of all blocks in this database.</param>
    Database(const char* databaseName, int32_t blockSize = 1 << 18);

    ~Database();

    void SetName(const std::string& newDatabaseName);

    const std::string& GetName() const
    {
        return name_;
    }

    /// <summary>
    /// Returns the deault database's block size. This blockSize is used (it's value), when creating
    /// a new table which does not have a specified blockSize.
    /// </summary>
    int GetBlockSize() const
    {
        return blockSize_;
    }

    std::unordered_map<std::string, Table>& GetTables()
    {
        return tables_;
    }

    static bool Exists(const std::string& databaseName)
    {
        return Context::getInstance().GetLoadedDatabases().find(databaseName) !=
               Context::getInstance().GetLoadedDatabases().end();
    }

    /// <summary>
    /// Returnes names of the loaded databases in memory.
    /// </summary>
    static std::vector<std::string> GetDatabaseNames();

    /// <summary>
    /// Copy block of data from the specified column of the source table and insert it in the column with the same name in the destination table.
    /// This is used when block sizes of those two tables are different.
    /// </summary>
    /// <param name="srcTable">Source table.</param>
    /// <param name="destTable">Destination table.</param>
    /// <param name="columnName">Name of the column which data will be copied.</param>
    static void CopyBlocksOfColumn(Table& srcTable, Table& destTable, const std::string& columnName);

    /// <summary>
    /// Save modified blocks of data of all loaded database to disk.
    /// </summary>
    static void SaveModifiedToDisk();

    /// <summary>
    /// Save only DB_EXTENSION file to disk into directory defined in configuration file.
    /// </summary>
    void PersistOnlyDbFile();

    /// <summary>
    /// Save database from memory to disk. Rewrites all database files.
    /// </summary>
    void Persist();

    /// <summary>
    /// Save modified blocks of the specified table from memory to disk.
    /// </summary>
    /// <param name="tableName">Name of the table which modified blocks of data will be saved.</param>
    void PersistOnlyModified(const std::string tableName);

    /// <summary>
    /// Save all databases currently loaded in memory to disk. Rewrites all loaded databases' files.
    /// </summary>
    static void SaveAllToDisk();

    /// <summary>
    /// Load databases from disk storage.
    /// </summary>
    static void LoadDatabasesFromDisk();

    /// <summary>
    /// Rename specified table.
    /// </summary>
    /// <param name="oldTableName">Table name of the table which will be renamed.</param>
    /// <param name="oldTableName">New table name to which the table will be renamed.</param>
    void RenameTable(const std::string& oldTablename, const std::string& newTableName);

    /// <summary>
    /// Delete database from disk. Deletes DB_EXTENSION and COLUMN_DATA_EXTENSION files which belong
    /// to the specified database. Database is not deleted from memory.
    /// </summary>
    void DeleteDatabaseFromDisk();

    /// <summary>
    /// <param name="tableName">Name of the table to be deleted.</param>
    /// Delete table from disk. Deletes COLUMN_DATA_EXTENSION files which belong to the specified
    /// table of currently loaded database. To alter DB_EXTENSION file, this action also calls a
    /// function PersistOnlyDbFile(). Table needs to be deleted from memory before calling this
    /// method, so that DB_EXTENSION file can be updated correctly.
    /// </summary>
    void DeleteTableFromDisk(const char* tableName);

    /// <summary>
    /// <param name="tableName">Name of the table which have the specified column that will be
    /// deleted.</param> <param name="columnName">Name of the COLUMN_DATA_EXTENSION file without the
    /// COLUMN_DATA_EXTENSION suffix that will be deleted.</param> Delete column of a table. Deletes
    /// single COLUMN_DATA_EXTENSION file which belongs to specified column and specified table. To
    /// alter DB_EXTENSION file, this action also calls a function Persist. Column needs to be deleted
    /// from memory before calling this method, so that DB_EXTENSION file can be updated correctly.
    /// </summary>
    void DeleteColumnFromDisk(const char* tableName, const char* columnName);

    /// <summary>
    /// Changes the block size of all tables of database and all the columns of the all tables will be affected - their blocks
    /// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
    /// </summary>
    /// <param name="newBlockSize">New block size of all tables and columns of this database.</param>
    void ChangeDatabaseBlockSize(int32_t newBlockSize);

    /// <summary>
    /// Changes the block size of a table and all the columns of the table will be affected - their blocks
    /// will be saved again with a bigger block size and the columns saved on disk of this table will be removed.
    /// </summary>
    /// <param name="tableName">Name of the table which have which block size will be changed.</param>
    /// <param name="newBlockSize">New block size of a table and columns of this table.</param>
    void ChangeTableBlockSize(const std::string tableName, int32_t newBlockSize);

    /// <summary>
    /// Load database from disk into memory.
    /// </summary>
    /// <param name="fileDbName">Name of the database file without the DB_EXTENSION suffix.</param>
    /// <param name="path">Path to directory in which database files are.</param>
    /// <returns>Newly created table.</returns>
    static std::shared_ptr<Database> LoadDatabase(const char* fileDbName, const char* path);

    /// <summary>
    /// Creates table with given name and columns and adds it to database. If the table already
    /// existed, create missing columns if there are any missing
    /// </summary>
    /// <param name="columns">Columns with types.</param>
    /// <param name="tableName">Table name.</param>
    /// <param name="areNullable">Nullablity of columns. Default values are set to be true.</param>
    /// <param name="blockSize">Table block size.</param>
    /// <returns>Newly created table.</returns>
    Table&
    CreateTable(const std::unordered_map<std::string, DataType>& columns,
                const char* tableName,
                const std::unordered_map<std::string, bool>& areNullable = std::unordered_map<std::string, bool>(),
                const std::unordered_map<std::string, bool>& areUnique = std::unordered_map<std::string, bool>(),
                int32_t blockSize = -1);

    /// <summary>
    /// Add database to in memory list.
    /// </summary>
    /// <param name="database">Database to be added.</param>
    static void AddToInMemoryDatabaseList(std::shared_ptr<Database> database);

    /// <summary>
    /// Get database from in memory list.
    /// </summary>
    /// <param name="databaseName">Name of database to get.</param>
    /// <returns>Database object or null-</returns>
    static std::shared_ptr<Database> GetDatabaseByName(std::string databaseName)
    {
        std::lock_guard<std::mutex> lock(dbAccessMutex_);
        try
        {
            return Context::getInstance().GetLoadedDatabases().at(databaseName);
        }
        catch (std::out_of_range&)
        {
            return nullptr;
        }
    }

    /// <summary>
    /// Remove database from in memory database list.
    /// </summary>
    /// <param name="databaseName">Name of database to be removed.</param>
    static void RemoveFromInMemoryDatabaseList(const char* databaseName);
};
