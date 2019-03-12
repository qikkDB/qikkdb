#include "Table.h"
#include "Database.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include "ColumnBase.h"
#include <cstdint>

/*#ifndef __CUDACC__
void Table::InsertValuesInNonIndexColumns(const std::unordered_map<std::string, std::any>& data, int indexBlock, int indexInBlock, std::string sortingColumn, int iterator, int range)
{
	for (const auto& column : columns)
	{
		const std::string columnName = column.first;
		auto search = data.find(columnName);
		if (search != data.end() && columnName != sortingColumn)
		{
			const auto &wrappedData = data.at(columnName);
			if (wrappedData.type() == typeid(std::vector<int32_t>))
			{
				std::vector<int32_t> dataInt = std::any_cast<std::vector<int32_t>>(wrappedData);
				dynamic_cast<ColumnBase<int32_t>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataInt[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<int64_t>))
			{
				std::vector<int64_t> dataLong = std::any_cast<std::vector<int64_t>>(wrappedData);
				dynamic_cast<ColumnBase<int64_t>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataLong[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<double>))
			{
				std::vector<double> dataDouble = std::any_cast<std::vector<double>>(wrappedData);
				dynamic_cast<ColumnBase<double>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataDouble[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<float>))
			{
				std::vector<float> dataFloat = std::any_cast<std::vector<float>>(wrappedData);
				dynamic_cast<ColumnBase<float>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataFloat[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<std::string>))
			{
				std::vector<std::string> dataString = std::any_cast<std::vector<std::string>>(wrappedData);
				dynamic_cast<ColumnBase<std::string>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataString[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
			{
				std::vector<ColmnarDB::Types::ComplexPolygon> dataPolygon = std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedData);
				dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataPolygon[iterator], range);
			}
			else if (wrappedData.type() == typeid(std::vector<ColmnarDB::Types::Point>))
			{
				std::vector<ColmnarDB::Types::Point> dataPoint = std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedData);
				dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(columns.find(columnName)->second.get())->InsertDataOnSpecificPosition(indexBlock, indexInBlock, dataPoint[iterator], range);
			}
		}
	}
} 
#endif*/

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
std::tuple<int, int, int>
Table::FindIndexAccordingPrimaryIndex(int index,
                                      int indexBlock,
                                      int indexInBlock,
                                      int range,
									  const std::unordered_map<std::string, std::any>& data)
{
    int columnIndexBlock = indexBlock;
    int columnIndexInBlock = indexInBlock;
    int columnRange = range;

    for (int j = 1; j < sortingColumns.size(); j++)
    {
        auto secondaryIndexedColumn = (columns.find(sortingColumns[j])->second.get());
        const auto& wrappedDataSecondaryIndexedColumn = data.at(sortingColumns[j]);

		if (wrappedDataSecondaryIndexedColumn.type() == typeid(std::vector<int32_t>))
        {
            std::vector<int32_t> dataIndexedColumn =std::any_cast<std::vector<int32_t>>(wrappedDataSecondaryIndexedColumn);
            auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(secondaryIndexedColumn);
            std::tie(columnIndexBlock, columnIndexInBlock, columnRange) =
                castedColumn->FindIndexAccordingPrimaryIndex(columnIndexInBlock, columnIndexBlock, columnRange, dataIndexedColumn[index]);
		}
		//TODO ostatne typy
    }

    return std::make_tuple(columnIndexBlock, columnIndexInBlock, columnRange);
}
#endif


#ifndef __CUDACC__
void Table::InsertData(const std::unordered_map<std::string, std::any>& data)
{
	int range = INT_MAX;
	if (!sortingColumns.empty())
	{
		for (auto sortingColumn : sortingColumns) 
		{
			auto primaryIndexedColumn = (columns.find(sortingColumn)->second.get());
			const auto &wrappedDataPrimaryIndexedColumn = data.at(sortingColumn);

			int indexBlock;
			int indexInBlock;

			if (wrappedDataPrimaryIndexedColumn.type() == typeid(std::vector<int32_t>))
			{
				std::vector<int32_t> dataIndexedColumn = std::any_cast<std::vector<int32_t>>(wrappedDataPrimaryIndexedColumn);
                auto castedColumn = dynamic_cast<ColumnBase<int32_t>*>(primaryIndexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
                   // std::tie(indexBlock, indexInBlock, range) = castedColumn->FindBlockIndexAndRange(dataIndexedColumn[i]);
					std::tie(indexBlock, indexInBlock, range) = FindIndexAccordingPrimaryIndex(i, indexBlock, indexInBlock, range, data);
				}
				//TODO insertni na spravnu poziciu
			}
			//TODO ostatne typy
			/*else if (wrappedDataIndexedColumn.type() == typeid(std::vector<int64_t>))
			{
				std::vector<int64_t> dataIndexedColumn = std::any_cast<std::vector<int64_t>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<int64_t>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock, range) = castedColumn->InsertOneValueData(dataIndexedColumn[i], range);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i, range);
				}
			}
			else if (wrappedDataIndexedColumn.type() == typeid(std::vector<double>))
			{
				std::vector<double> dataIndexedColumn = std::any_cast<std::vector<double>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<double>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock, range) = castedColumn->InsertOneValueData(dataIndexedColumn[i], range);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i, range);
				}
			}
			else if (wrappedDataIndexedColumn.type() == typeid(std::vector<float>))
			{
				std::vector<float> dataIndexedColumn = std::any_cast<std::vector<float>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<float>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock, range) = castedColumn->InsertOneValueData(dataIndexedColumn[i], range);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i, range);
				}
			}*/
			//TODO Point and Polygon and string indexes (decide what is the minimum of these and decide how compare them)
			/*else if (wrappedDataIndexedColumn.type() == typeid(std::vector<std::string>))
			{
				std::vector<std::string> dataIndexedColumn = std::any_cast<std::vector<std::string>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<std::string>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock) = castedColumn->InsertOneValueData(dataIndexedColumn[i]);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i);
				}
			}
			else if (wrappedDataIndexedColumn.type() == typeid(std::vector<ColmnarDB::Types::ComplexPolygon>))
			{
				std::vector<ColmnarDB::Types::ComplexPolygon> dataIndexedColumn = std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock) = castedColumn->InsertOneValueData(dataIndexedColumn[i]);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i);
				}
			}
			else if (wrappedDataIndexedColumn.type() == typeid(std::vector<ColmnarDB::Types::Point>))
			{
				std::vector<ColmnarDB::Types::Point> dataIndexedColumn = std::any_cast<std::vector<ColmnarDB::Types::Point>>(wrappedDataIndexedColumn);
				auto castedColumn = dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(indexedColumn);

				for (int i = 0; i < dataIndexedColumn.size(); i++)
				{
					std::tie(indexBlock, indexInBlock) = castedColumn->InsertOneValueData(dataIndexedColumn[i]);
					InsertValuesInNonIndexColumns(data, indexBlock, indexInBlock, sortingColumn, i);
				}
			}*/
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
