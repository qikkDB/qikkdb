#include "GpuSqlDispatcherOrderByFunctions.h"

#include <vector>
#include <limits>
#include <cstdint>

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
	// Tieto polia nie su uniformne velke !!!, treba to zistit z IVariantArray
	std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByOrderColumnBlocks;
	std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByRetColumnBlocks;
	std::unordered_map<std::string, std::unique_ptr<IVariantArray>> reconstructedOrderByColumnsMerged;
	std::unordered_map<int32_t, std::pair<std::string, OrderBy::Order>> orderByColumns;
	*/

	// Allocate a vector of merge pointers to the input vectors - counters that hold the merge positions - initialize them to zero
	// Allocate a vector that holds the sizes of the input blocks - the length of this vector equals to the number of input blocks
	int32_t blockCount = reconstructedOrderByRetColumnBlocks.begin()->second.size();
	std::vector<int32_t> merge_counters(blockCount, 0);
	std::vector<int32_t> merge_limits(blockCount);

	for(int32_t i = 0; i < blockCount; i++)
	{
		merge_limits.push_back(reconstructedOrderByRetColumnBlocks.begin()->second[i]->GetSize());
	}

	// Merge the input arrays to the output arrays
	bool arraysMerged = false;
	while(arraysMerged != true)
	{
		// A flag for checking if a value for insetion was found
		bool valueToInsertFound = false;
		
		// Check each entry from left to right (the numbers are in inverse because of the dispatcher)
		for(int32_t i = orderByColumns.size() - 1, j = 0; i >= 0; i--, j++)
		{
			// If given column is ASC - find a global minimum
			// else if given column is DESC - find a global maximum
			if(orderByColumns[i].second == OrderBy::Order::ASC)
			{
				// Find global minimum - neeed to distinguish between different data types
				int32_t minimumBlockIdx = -1;
				int32_t minimum = std::numeric_limits<int32_t>::max();
				for(int32_t k = 0; k < merge_counters.size(); k++)
				{
					// Check if we are within the merged block sizes
					if(merge_counters[] < merge_limits[]) {
						reconstructedOrderByOrderColumnBlocks[orderByColumns[i].first];
					}
				}
				


				//reconstructedOrderByRetColumnBlocks[orderByColumns[i].first][];


				valueToInsertFound = true;
			}
			else if(orderByColumns[i].second == OrderBy::Order::DESC)
			{
				
				valueToInsertFound = true;
			}
		}

		// If no value to insert was found - insert the first availiable
	}


	return 0;
}
