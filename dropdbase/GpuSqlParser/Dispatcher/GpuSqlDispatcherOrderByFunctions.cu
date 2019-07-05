#include "GpuSqlDispatcherOrderByFunctions.h"

#include <vector>
#include <limits>
#include <cstdint>

#include "../../DataType.h"
#include "../../VariantArray.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByFunctions = { &GpuSqlDispatcher::orderByConst<int32_t>, &GpuSqlDispatcher::orderByConst<int64_t>, &GpuSqlDispatcher::orderByConst<float>, &GpuSqlDispatcher::orderByConst<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &GpuSqlDispatcher::orderByConst<int8_t>, &GpuSqlDispatcher::orderByCol<int32_t>, &GpuSqlDispatcher::orderByCol<int64_t>, &GpuSqlDispatcher::orderByCol<float>, &GpuSqlDispatcher::orderByCol<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &GpuSqlDispatcher::orderByCol<int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByReconstructOrderFunctions = { &GpuSqlDispatcher::orderByReconstructOrderConst<int32_t>, &GpuSqlDispatcher::orderByReconstructOrderConst<int64_t>, &GpuSqlDispatcher::orderByReconstructOrderConst<float>, &GpuSqlDispatcher::orderByReconstructOrderConst<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &GpuSqlDispatcher::orderByReconstructOrderConst<int8_t>, &GpuSqlDispatcher::orderByReconstructOrderCol<int32_t>, &GpuSqlDispatcher::orderByReconstructOrderCol<int64_t>, &GpuSqlDispatcher::orderByReconstructOrderCol<float>, &GpuSqlDispatcher::orderByReconstructOrderCol<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &GpuSqlDispatcher::orderByReconstructOrderCol<int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByReconstructRetFunctions = { &GpuSqlDispatcher::orderByReconstructRetConst<int32_t>, &GpuSqlDispatcher::orderByReconstructRetConst<int64_t>, &GpuSqlDispatcher::orderByReconstructRetConst<float>, &GpuSqlDispatcher::orderByReconstructRetConst<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &GpuSqlDispatcher::orderByReconstructRetConst<int8_t>, &GpuSqlDispatcher::orderByReconstructRetCol<int32_t>, &GpuSqlDispatcher::orderByReconstructRetCol<int64_t>, &GpuSqlDispatcher::orderByReconstructRetCol<float>, &GpuSqlDispatcher::orderByReconstructRetCol<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &GpuSqlDispatcher::orderByReconstructRetCol<int8_t> };

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::freeOrderByTableFunction = &GpuSqlDispatcher::freeOrderByTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::orderByReconstructRetAllBlocksFunction = &GpuSqlDispatcher::orderByReconstructRetAllBlocks;

int32_t GpuSqlDispatcher::freeOrderByTable()
{
	std::cout << "Freeing order by table." << std::endl;
	orderByTable.release();
	return 0;
}

int32_t GpuSqlDispatcher::orderByReconstructRetAllBlocks()
{

	/*
	switch(.GetType())
	{
		case COLUMN_INT:
		case COLUMN_LONG:
		case COLUMN_FLOAT:
		case COLUMN_DOUBLE:
		case COLUMN_POINT:
		case COLUMN_POLYGON:
		case COLUMN_STRING:
		case COLUMN_INT8_T:
	}
	*/

	if(isOverallLastBlock)
	{
		// Count and allocate the result vectors for the output map
		int32_t resultSetSize = 0;
		int32_t resultSetCounter = 0;

		// Allocate a vector of merge pointers to the input vectors - counters that hold the merge positions - initialize them to zero
		// Allocate a vector that holds the sizes of the input blocks - the length of this vector equals to the number of input blocks
		int32_t blockCount = reconstructedOrderByOrderColumnBlocks.begin()->second.size();
		std::vector<int32_t> merge_counters(blockCount, 0);
		std::vector<int32_t> merge_limits(blockCount);

		for(int32_t i = 0; i < blockCount; i++)
		{
			int32_t blockSize = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
			resultSetSize += blockSize;
			merge_limits[i] = blockSize;
		}

		// Allocate the result map by inserting a column name and iVariantArray pair
		for(auto &orderColumn : orderByColumns)
		{
			reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<int32_t>>(resultSetSize);
		}

		//Write the results to the result map
		bool dataMerged = false;
		while(dataMerged != true)
		{
			// Merge the input arrays to the output arrays
			// Check each entry from left to right (the numbers are in inverse because of the dispatcher)
			for(int32_t i = orderByColumns.size() - 1; i >= 0; i--)
			{
				// Check if all values pointed to by the counters are equal, if yes - proceed to the next column
				bool valuesAreEqual = true;
				int32_t firstNonzeroMergeCounterIdx = -1;
				int32_t lastValue = -1;
				for(int32_t j = 0; j < merge_counters.size(); j++)
				{
					if(lastValue == -1 && merge_counters[j] < merge_limits[j]) {
						lastValue = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
						firstNonzeroMergeCounterIdx = j;
					}
					else if (merge_counters[j] < merge_limits[j]) {
						int32_t value = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
						if(lastValue != value)
						{
							valuesAreEqual = false;
							break;
						}
					}
				}

				// If no first nonzero index was found - there are no entries left - terminate the while loop
				if(firstNonzeroMergeCounterIdx == -1)
				{
					dataMerged = true;
					break;
				}

				// If all values in the valid merge pointers are equal - continue to the next column
				// If this column is the last column - insert the next value and exit the loop
				if(valuesAreEqual && i > 0)
				{
					continue;
				}
				else if(valuesAreEqual && i == 0)
				{
					// Instert a tuple at first nonzero place and break
					if(resultSetCounter < resultSetSize)
					{
						// The program copies the result values - based on column name
						for(auto &retColumn : reconstructedOrderByRetColumnBlocks)
						{
							int32_t value = dynamic_cast<VariantArray<int32_t>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
							dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
						}
						resultSetCounter++;
					}
					else {
						//result set is full - need to break the while cycle - THIS MAY BE FAULTY !!!
						dataMerged = true;
						break;
					}

					merge_counters[firstNonzeroMergeCounterIdx]++;
					break;
				}

				// If values are not equal
				// If given column is ASC - find a global minimum
				// else if given column is DESC - find a global maximum
				// Find global minimum or maximum depending on the column type - neeed to distinguish between different data types
				
				int32_t mergeCounterIdx = -1;
				int32_t minimum = std::numeric_limits<int32_t>::max();
				int32_t maximum = std::numeric_limits<int32_t>::lowest();

				for(int32_t j = 0; j < merge_counters.size(); j++)
				{
					// Check if we are within the merged block sizes
					if(orderByColumns[i].second == OrderBy::Order::ASC && merge_counters[j] < merge_limits[j]) {
						// Get the value from the block to which the merge counter points
						int32_t value = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
						if(minimum > value)
						{
							minimum = value;
							mergeCounterIdx = j;
						}
					}
					else if(orderByColumns[i].second == OrderBy::Order::DESC && merge_counters[j] < merge_limits[j]) {
						// Get the value from the block to which the merge counter points
						int32_t value = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
						if(maximum < value)
						{
							maximum = value;
							mergeCounterIdx = j;
						}
					}
				}

				// If an extrem was found (min or max)
				if(mergeCounterIdx != -1)
				{
					// Insert and break
					if(resultSetCounter < resultSetSize)
					{
						// The program copies the result values - based on column name
						int sz = 0;
						for(auto &retColumn : reconstructedOrderByRetColumnBlocks)
						{
							int32_t value = dynamic_cast<VariantArray<int32_t>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
							dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;

						}
						resultSetCounter++;
					}
					else {
						//result set is full - need to break the while cycle - THIS MAY BE FAULTY !!!
						dataMerged = true;
					}

					merge_counters[mergeCounterIdx]++;
					break;
				}
				else {
					// ???
				}
			}
		}
	}
	return 0;
}
