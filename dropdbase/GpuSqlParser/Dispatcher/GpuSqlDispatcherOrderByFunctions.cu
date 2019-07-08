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
			int32_t blockSize = 0;

			// Retrieve the variant array type
			switch(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get()->GetType())
			{
				case COLUMN_INT:
					blockSize = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
					break;
				case COLUMN_LONG:
					blockSize = dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
					break;
				case COLUMN_FLOAT:
					blockSize = dynamic_cast<VariantArray<float>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
					break;
				case COLUMN_DOUBLE:
					blockSize = dynamic_cast<VariantArray<double>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
					break;
				case COLUMN_POINT:
					throw std::runtime_error("ORDER BY operation not implemented for points");
				case COLUMN_POLYGON:
					throw std::runtime_error("ORDER BY operation not implemented for polygons");
				case COLUMN_STRING:
					throw std::runtime_error("ORDER BY operation not implemented for strings");
				case COLUMN_INT8_T:
					blockSize = dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())->GetSize();
					break;
				default:
					break;
			}

			resultSetSize += blockSize;
			merge_limits[i] = blockSize;
		}

		// Allocate the result map by inserting a column name and iVariantArray pair
		for(auto &orderColumn : orderByColumns)
		{
			// Retrieve the variant array type of the return columns - WARNING - this works only for non empty columns
			switch(reconstructedOrderByRetColumnBlocks[orderColumn.second.first][0].get()->GetType())
			{
				case COLUMN_INT:
					reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<int32_t>>(resultSetSize);
					break;
				case COLUMN_LONG:
					reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<int64_t>>(resultSetSize);
					break;
				case COLUMN_FLOAT:
					reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<float>>(resultSetSize);
					break;
				case COLUMN_DOUBLE:
					reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<double>>(resultSetSize);
					break;
				case COLUMN_POINT:
					throw std::runtime_error("ORDER BY operation not implemented for points");
				case COLUMN_POLYGON:
					throw std::runtime_error("ORDER BY operation not implemented for polygons");
				case COLUMN_STRING:
					throw std::runtime_error("ORDER BY operation not implemented for strings");
				case COLUMN_INT8_T:
					reconstructedOrderByColumnsMerged[orderColumn.second.first] = std::make_unique<VariantArray<int8_t>>(resultSetSize);
					break;
				default:
					break;
			}
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
				bool lastValueFound = true;
				int32_t firstNonzeroMergeCounterIdx = -1;

				// WARNING - this works only for non empty columns
				switch(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][0].get()->GetType())
				{
					case COLUMN_INT: 
						{
							int32_t lastValue = 0;
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								if(lastValueFound && merge_counters[j] < merge_limits[j]) {
									lastValue = dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									firstNonzeroMergeCounterIdx = j;
									lastValueFound = false;
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
						}
						break;
					case COLUMN_LONG:
						{
							int64_t lastValue = 0;
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								if(lastValueFound && merge_counters[j] < merge_limits[j]) {
									lastValue = dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									firstNonzeroMergeCounterIdx = j;
									lastValueFound = false;
								}
								else if (merge_counters[j] < merge_limits[j]) {
									int64_t value = dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(lastValue != value)
									{
										valuesAreEqual = false;
										break;
									}
								}
							}
						}
						break;
					case COLUMN_FLOAT:
						{
							float lastValue = 0;
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								if(lastValueFound && merge_counters[j] < merge_limits[j]) {
									lastValue = dynamic_cast<VariantArray<float>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									firstNonzeroMergeCounterIdx = j;
									lastValueFound = false;
								}
								else if (merge_counters[j] < merge_limits[j]) {
									float value = dynamic_cast<VariantArray<float>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(lastValue != value)
									{
										valuesAreEqual = false;
										break;
									}
								}
							}
						}
						break;
					case COLUMN_DOUBLE:
						{
							double lastValue = 0;
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								if(lastValueFound && merge_counters[j] < merge_limits[j]) {
									lastValue = dynamic_cast<VariantArray<double>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									firstNonzeroMergeCounterIdx = j;
									lastValueFound = false;
								}
								else if (merge_counters[j] < merge_limits[j]) {
									double value = dynamic_cast<VariantArray<double>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(lastValue != value)
									{
										valuesAreEqual = false;
										break;
									}
								}
							}
						}
						break;
					case COLUMN_POINT:
						throw std::runtime_error("ORDER BY operation not implemented for points");
					case COLUMN_POLYGON:
						throw std::runtime_error("ORDER BY operation not implemented for polygons");
					case COLUMN_STRING:
						throw std::runtime_error("ORDER BY operation not implemented for strings");
					case COLUMN_INT8_T:
						{
							int8_t lastValue = 0;
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								if(lastValueFound && merge_counters[j] < merge_limits[j]) {
									lastValue = dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									firstNonzeroMergeCounterIdx = j;
									lastValueFound = false;
								}
								else if (merge_counters[j] < merge_limits[j]) {
									int8_t value = dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(lastValue != value)
									{
										valuesAreEqual = false;
										break;
									}
								}
							}
						}
						break;
					default:
						break;
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
							switch(retColumn.second[0].get()->GetType())
							{
								case COLUMN_INT:
									{
										int32_t value = dynamic_cast<VariantArray<int32_t>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
										dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_LONG:
									{
										int64_t value = dynamic_cast<VariantArray<int64_t>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
										dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_FLOAT:
									{
										float value = dynamic_cast<VariantArray<float>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
										dynamic_cast<VariantArray<float>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_DOUBLE:
									{
										double value = dynamic_cast<VariantArray<double>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
										dynamic_cast<VariantArray<double>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_POINT:
									throw std::runtime_error("ORDER BY operation not implemented for points");
								case COLUMN_POLYGON:
									throw std::runtime_error("ORDER BY operation not implemented for polygons");								
								case COLUMN_STRING:
									throw std::runtime_error("ORDER BY operation not implemented for strings");									
								case COLUMN_INT8_T:
									{
										int8_t value = dynamic_cast<VariantArray<int8_t>*>(retColumn.second[firstNonzeroMergeCounterIdx].get())->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
										dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								default:
									break;
							}
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

				// WARNING - this works only for non empty columns
				switch(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][0].get()->GetType())
				{
					case COLUMN_INT:
						{
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
						}
						break;
					case COLUMN_LONG:
						{
							int64_t minimum = std::numeric_limits<int64_t>::max();
							int64_t maximum = std::numeric_limits<int64_t>::lowest();
			
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								// Check if we are within the merged block sizes
								if(orderByColumns[i].second == OrderBy::Order::ASC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									int64_t value = dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(minimum > value)
									{
										minimum = value;
										mergeCounterIdx = j;
									}
								}
								else if(orderByColumns[i].second == OrderBy::Order::DESC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									int64_t value = dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(maximum < value)
									{
										maximum = value;
										mergeCounterIdx = j;
									}
								}
							}
						}
						break;
					case COLUMN_FLOAT:
						{
							float minimum = std::numeric_limits<float>::max();
							float maximum = std::numeric_limits<float>::lowest();
			
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								// Check if we are within the merged block sizes
								if(orderByColumns[i].second == OrderBy::Order::ASC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									float value = dynamic_cast<VariantArray<float>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(minimum > value)
									{
										minimum = value;
										mergeCounterIdx = j;
									}
								}
								else if(orderByColumns[i].second == OrderBy::Order::DESC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									float value = dynamic_cast<VariantArray<float>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(maximum < value)
									{
										maximum = value;
										mergeCounterIdx = j;
									}
								}
							}
						}
						break;
					case COLUMN_DOUBLE:
						{
							double minimum = std::numeric_limits<double>::max();
							double maximum = std::numeric_limits<double>::lowest();
			
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								// Check if we are within the merged block sizes
								if(orderByColumns[i].second == OrderBy::Order::ASC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									double value = dynamic_cast<VariantArray<double>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(minimum > value)
									{
										minimum = value;
										mergeCounterIdx = j;
									}
								}
								else if(orderByColumns[i].second == OrderBy::Order::DESC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									double value = dynamic_cast<VariantArray<double>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(maximum < value)
									{
										maximum = value;
										mergeCounterIdx = j;
									}
								}
							}
						}
						break;
					case COLUMN_POINT:
						throw std::runtime_error("ORDER BY operation not implemented for points");
					case COLUMN_POLYGON:
						throw std::runtime_error("ORDER BY operation not implemented for polygons");								
					case COLUMN_STRING:
						throw std::runtime_error("ORDER BY operation not implemented for strings");									
					case COLUMN_INT8_T:
						{
							int8_t minimum = std::numeric_limits<int8_t>::max();
							int8_t maximum = std::numeric_limits<int8_t>::lowest();
			
							for(int32_t j = 0; j < merge_counters.size(); j++)
							{
								// Check if we are within the merged block sizes
								if(orderByColumns[i].second == OrderBy::Order::ASC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									int8_t value = dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(minimum > value)
									{
										minimum = value;
										mergeCounterIdx = j;
									}
								}
								else if(orderByColumns[i].second == OrderBy::Order::DESC && merge_counters[j] < merge_limits[j]) {
									// Get the value from the block to which the merge counter points
									int8_t value = dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first][j].get())->getData()[merge_counters[j]];
									if(maximum < value)
									{
										maximum = value;
										mergeCounterIdx = j;
									}
								}
							}
						}
						break;
					default:
						break;
				}

				// If an extrem was found (min or max)
				if(mergeCounterIdx != -1)
				{
					// Insert and break
					if(resultSetCounter < resultSetSize)
					{
						// The program copies the result values - based on column name
						for(auto &retColumn : reconstructedOrderByRetColumnBlocks)
						{
							switch(retColumn.second[0].get()->GetType())
							{
								case COLUMN_INT:
									{
										int32_t value = dynamic_cast<VariantArray<int32_t>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
										dynamic_cast<VariantArray<int32_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_LONG:
									{
										int64_t value = dynamic_cast<VariantArray<int64_t>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
										dynamic_cast<VariantArray<int64_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_FLOAT:
									{
										float value = dynamic_cast<VariantArray<float>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
										dynamic_cast<VariantArray<float>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_DOUBLE:
									{
										double value = dynamic_cast<VariantArray<double>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
										dynamic_cast<VariantArray<double>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								case COLUMN_POINT:
									throw std::runtime_error("ORDER BY operation not implemented for points");
								case COLUMN_POLYGON:
									throw std::runtime_error("ORDER BY operation not implemented for polygons");								
								case COLUMN_STRING:
									throw std::runtime_error("ORDER BY operation not implemented for strings");									
								case COLUMN_INT8_T:
									{
										int8_t value = dynamic_cast<VariantArray<int8_t>*>(retColumn.second[mergeCounterIdx].get())->getData()[merge_counters[mergeCounterIdx]];
										dynamic_cast<VariantArray<int8_t>*>(reconstructedOrderByColumnsMerged[retColumn.first].get())->getData()[resultSetCounter] = value;
									}
									break;
								default:
									break;
							}
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
