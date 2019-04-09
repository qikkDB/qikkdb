#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include <cstdint>


#ifndef __CUDACC__
void Table::InsertValuesOnSpecificPosition(const std::unordered_map<std::string, std::any>& data, int indexBlock, int indexInBlock, int iterator)
{
    for (const auto& column : columns)
    {
        const std::string columnName = column.first;
        auto search = data.find(columnName);
        if (search != data.end())
        {
            auto currentColumn = (columns.find(columnName)->second.get());
            const auto &wrappedData = data.at(columnName);

            if (wrappedData.type() == typeid(std::vector<int32_t>))
            {
                std::vector<int32_t> dataInt = std::any_cast<std::vector<int32_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataInt[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<int64_t>))
            {
                std::vector<int64_t> dataLong = std::any_cast<std::vector<int64_t>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataLong[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<double>))
            {
                std::vector<double> dataDouble = std::any_cast<std::vector<double>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataDouble[iterator]);
            }
            else if (wrappedData.type() == typeid(std::vector<float>))
            {
                std::vector<float> dataFloat = std::any_cast<std::vector<float>>(wrappedData);
                auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentColumn);
                castedColumn->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataFloat[iterator]);
            }
			//TODO string, point, polygon
        }
    }
}
#endif

#ifndef __CUDACC__
int32_t Table::getDataSizeOfInsertedColumns(const std::unordered_map<std::string, std::any>& data)
{
    int size;

    auto firstSortingColumn = (columns.find(sortingColumns[0])->second.get());
    const auto& dataOfFirstColumn = data.at(sortingColumns[0]);

    if (dataOfFirstColumn.type() == typeid(std::vector<int32_t>))
    {
        std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
	}

	if (dataOfFirstColumn.type() == typeid(std::vector<int64_t>))
    {
        std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	if (dataOfFirstColumn.type() == typeid(std::vector<double>))
    {
        std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	if (dataOfFirstColumn.type() == typeid(std::vector<float>))
    {
        std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(dataOfFirstColumn);
        size = dataIndexedColumn.size();
    }

	//TODO for polygon, point and string

		return size;
}
#endif

const std::shared_ptr<Database>& Table::GetDatabase()
{
	return database;
}

const std::string & Table::GetName() const
{
	return name;
}

int Table::GetBlockSize() const
{
	return blockSize;
}

int32_t Table::GetBlockCount() const
{
	for (auto& column : columns) 
	{
		return column.second.get()->GetBlockCount();
	}
	return 0;
}

const std::unordered_map<std::string, std::unique_ptr<IColumn>>& Table::GetColumns() const
{
	return columns;
}

std::vector<std::string> Table::GetSortingColumns()
{
	return sortingColumns;
}

void Table::SetSortingColumns(std::vector<std::string> columns)
{
	sortingColumns = columns;
}

Table::Table(const std::shared_ptr<Database> &database, const char* name) : database(database), name(name)
{
	blockSize = database->GetBlockSize();
}

void Table::CreateColumn(const char* columnName, DataType columnType)
{
	std::unique_ptr<IColumn> column;

	if (columnType == COLUMN_INT)
	{
		column = std::make_unique<ColumnBase<int32_t>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_LONG)
	{
		column = std::make_unique<ColumnBase<int64_t>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_DOUBLE)
	{
		column = std::make_unique<ColumnBase<double>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_FLOAT)
	{
		column = std::make_unique<ColumnBase<float>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_STRING)
	{
		column = std::make_unique<ColumnBase<std::string>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_POLYGON)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::ComplexPolygon>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_POINT)
	{
		column = std::make_unique<ColumnBase<ColmnarDB::Types::Point>>(columnName, blockSize);
	}
	else if (columnType == COLUMN_INT8_T)
	{
		column = std::make_unique<ColumnBase<int8_t>>(columnName, blockSize);
	}
	columns.insert(std::make_pair(columnName, std::move(column)));

	
}

#ifndef __CUDACC__
void Table::InsertData(const std::unordered_map<std::string, std::any>& data)
{
	if (!sortingColumns.empty())
	{
        int oneColumnDataSize = getDataSizeOfInsertedColumns(data);
        std::cout << oneColumnDataSize << std::endl;
		
		for (int i = 0; i < oneColumnDataSize; i++)
        {
            int range = INT_MAX;
            int blockIndex = 0;
            int indexInBlock = 0;

            for (auto sortingColumn : sortingColumns)
            {
                std::cout << sortingColumn << std::endl;
                auto currentSortingColumn = (columns.find(sortingColumn)->second.get());
                const auto& wrappedCurrentSortingColumnData = data.at(sortingColumn);

                if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int32_t>))
                {
                    std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);

					
					std::tie(blockIndex, indexInBlock, range) = castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                    std::cout << blockIndex << " " << indexInBlock << " " << range << " "
                              << " " << dataIndexedColumn[i] << std::endl;
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<int64_t>))
                {
                    std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<float>))
                {
                    std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<float>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				if (wrappedCurrentSortingColumnData.type() == typeid(std::vector<double>))
                {
                    std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(wrappedCurrentSortingColumnData);
                    auto castedColumn = dynamic_cast<ColumnBase<double>*>(currentSortingColumn);

                    std::tie(blockIndex, indexInBlock, range) =
                        castedColumn->FindIndexAndRange(blockIndex, indexInBlock, range, dataIndexedColumn[i]);
                }

				//TODO string, polygon, point
            }

			InsertValuesOnSpecificPosition(data,blockIndex,indexInBlock,i);
			
		}
	}

	else
	{
		for (const auto& column : columns)
		{
			std::string columnName = column.first;
			auto search = data.find(columnName);
			if (search != data.end())
			{
				const auto &wrappedData = data.at(columnName);
				if (wrappedData.type() == typeid(std::vector<int32_t>))
				{
					dynamic_cast<ColumnBase<int32_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int32_t>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<int64_t>))
				{
					dynamic_cast<ColumnBase<int64_t>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<int64_t>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<double>))
				{
					dynamic_cast<ColumnBase<double>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<double>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<float>))
				{
					dynamic_cast<ColumnBase<float>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<float>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<std::string>))
				{
					dynamic_cast<ColumnBase<std::string>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<std::string>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
				{
					dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData));
				}
				else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
				{
					dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columns.find(columnName)->second.get())->InsertData(std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData));
				}
			}
		}
	}
}
#endif

bool Table::ContainsColumn(const char* column)
{
	auto search = columns.find(column);
	if (search != columns.end()) {
		return true;
	}
	return false;
}
