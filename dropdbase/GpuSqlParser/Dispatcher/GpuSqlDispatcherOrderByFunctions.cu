#include "GpuSqlDispatcherOrderByFunctions.h"

#include <vector>
#include <limits>
#include <cstdint>

#include "../../DataType.h"
#include "../../VariantArray.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByFunctions_ = {
    &GpuSqlDispatcher::OrderByConst<int32_t>,
    &GpuSqlDispatcher::OrderByConst<int64_t>,
    &GpuSqlDispatcher::OrderByConst<float>,
    &GpuSqlDispatcher::OrderByConst<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string>,
    &GpuSqlDispatcher::OrderByConst<int8_t>,
    &GpuSqlDispatcher::OrderByCol<int32_t>,
    &GpuSqlDispatcher::OrderByCol<int64_t>,
    &GpuSqlDispatcher::OrderByCol<float>,
    &GpuSqlDispatcher::OrderByCol<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string>,
    &GpuSqlDispatcher::OrderByCol<int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByReconstructOrderFunctions_ = {
    &GpuSqlDispatcher::OrderByReconstructOrderConst<int32_t>,
    &GpuSqlDispatcher::OrderByReconstructOrderConst<int64_t>,
    &GpuSqlDispatcher::OrderByReconstructOrderConst<float>,
    &GpuSqlDispatcher::OrderByReconstructOrderConst<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string>,
    &GpuSqlDispatcher::OrderByReconstructOrderConst<int8_t>,
    &GpuSqlDispatcher::OrderByReconstructOrderCol<int32_t>,
    &GpuSqlDispatcher::OrderByReconstructOrderCol<int64_t>,
    &GpuSqlDispatcher::OrderByReconstructOrderCol<float>,
    &GpuSqlDispatcher::OrderByReconstructOrderCol<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string>,
    &GpuSqlDispatcher::OrderByReconstructOrderCol<int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByReconstructRetFunctions_ = {
    &GpuSqlDispatcher::OrderByReconstructRetConst<int32_t>,
    &GpuSqlDispatcher::OrderByReconstructRetConst<int64_t>,
    &GpuSqlDispatcher::OrderByReconstructRetConst<float>,
    &GpuSqlDispatcher::OrderByReconstructRetConst<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string>,
    &GpuSqlDispatcher::OrderByReconstructRetConst<int8_t>,
    &GpuSqlDispatcher::OrderByReconstructRetCol<int32_t>,
    &GpuSqlDispatcher::OrderByReconstructRetCol<int64_t>,
    &GpuSqlDispatcher::OrderByReconstructRetCol<float>,
    &GpuSqlDispatcher::OrderByReconstructRetCol<double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string>,
    &GpuSqlDispatcher::OrderByReconstructRetCol<int8_t>};

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::freeOrderByTableFunction_ = &GpuSqlDispatcher::FreeOrderByTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::orderByReconstructRetAllBlocksFunction_ =
    &GpuSqlDispatcher::OrderByReconstructRetAllBlocks;

int32_t GpuSqlDispatcher::FreeOrderByTable()
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Freeing order by table." << '\n';
    orderByTable_.release();
    return 0;
}

int32_t GpuSqlDispatcher::OrderByReconstructRetAllBlocks()
{
    if (!usingGroupBy_ && isLastBlockOfDevice_)
    {
        if (isOverallLastBlock_)
        {
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::orderByMutex_);
            GpuSqlDispatcher::orderByCV_.wait(lock, [] { return GpuSqlDispatcher::IsOrderByDone(); });

            CudaLogBoost::getInstance(CudaLogBoost::info) << "Merging partially ordered blocks." << '\n';

            std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByOrderColumnBlocks;
            std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByRetColumnBlocks;

            std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> reconstructedOrderByOrderColumnNullBlocks;
            std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> reconstructedOrderByRetColumnNullBlocks;

            for (int32_t i = 0; i < Context::getInstance().getDeviceCount(); i++)
            {
                for (auto& orderBlocks : orderByBlocks_[i].ReconstructedOrderByOrderColumnBlocks)
                {
                    for (auto& orderBlockArray : orderBlocks.second)
                    {
                        reconstructedOrderByOrderColumnBlocks[orderBlocks.first].push_back(std::move(orderBlockArray));
                    }
                }

                for (auto& retBlocks : orderByBlocks_[i].ReconstructedOrderByRetColumnBlocks)
                {
                    for (auto& retBlockArray : retBlocks.second)
                    {
                        reconstructedOrderByRetColumnBlocks[retBlocks.first].push_back(std::move(retBlockArray));
                    }
                }

                for (auto& orderBlocksNull : orderByBlocks_[i].ReconstructedOrderByOrderColumnNullBlocks)
                {
                    for (auto& orderBlockArray : orderBlocksNull.second)
                    {
                        reconstructedOrderByOrderColumnNullBlocks[orderBlocksNull.first].push_back(
                            std::move(orderBlockArray));
                    }
                }

                for (auto& retBlocksNull : orderByBlocks_[i].ReconstructedOrderByRetColumnNullBlocks)
                {
                    for (auto& retBlockArray : retBlocksNull.second)
                    {
                        reconstructedOrderByRetColumnNullBlocks[retBlocksNull.first].push_back(
                            std::move(retBlockArray));
                    }
                }
            }

            // Count and allocate the result vectors for the output map
            int32_t resultSetSize = 0;
            int32_t resultSetNullSize = 0;
            int32_t resultSetCounter = 0;

            // Allocate a vector of merge pointers to the input vectors - counters that hold the
            // merge positions - initialize them to zero Allocate a vector that holds the sizes of
            // the input blocks - the length of this vector equals to the number of input blocks
            int32_t blockCount = reconstructedOrderByOrderColumnBlocks.begin()->second.size();
            std::vector<int32_t> merge_counters(blockCount, 0);
            std::vector<int32_t> merge_limits(blockCount);

            for (int32_t i = 0; i < blockCount; i++)
            {
                int32_t blockSize = 0;

                // Retrieve the variant array type
                switch (reconstructedOrderByOrderColumnBlocks.begin()->second[i].get()->GetType())
                {
                case COLUMN_INT:
                    blockSize = dynamic_cast<VariantArray<int32_t>*>(
                                    reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())
                                    ->GetSize();
                    break;
                case COLUMN_LONG:
                    blockSize = dynamic_cast<VariantArray<int64_t>*>(
                                    reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())
                                    ->GetSize();
                    break;
                case COLUMN_FLOAT:
                    blockSize = dynamic_cast<VariantArray<float>*>(
                                    reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())
                                    ->GetSize();
                    break;
                case COLUMN_DOUBLE:
                    blockSize = dynamic_cast<VariantArray<double>*>(
                                    reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())
                                    ->GetSize();
                    break;
                case COLUMN_POINT:
                    throw std::runtime_error("ORDER BY operation not implemented for points");
                case COLUMN_POLYGON:
                    throw std::runtime_error("ORDER BY operation not implemented for polygons");
                case COLUMN_STRING:
                    throw std::runtime_error("ORDER BY operation not implemented for strings");
                case COLUMN_INT8_T:
                    blockSize = dynamic_cast<VariantArray<int8_t>*>(
                                    reconstructedOrderByOrderColumnBlocks.begin()->second[i].get())
                                    ->GetSize();
                    break;
                default:
                    break;
                }

                resultSetSize += blockSize;
                merge_limits[i] = blockSize;
            }

            resultSetNullSize = (resultSetSize + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);

            // Allocate the result map by inserting a column name and iVariantArray pair
            for (auto& orderColumn : reconstructedOrderByRetColumnBlocks)
            {
                // Retrieve the variant array type of the return columns - WARNING - this works only for non empty columns
                switch (orderColumn.second[0].get()->GetType())
                {
                case COLUMN_INT:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<int32_t>>(resultSetSize);
                    break;
                case COLUMN_LONG:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<int64_t>>(resultSetSize);
                    break;
                case COLUMN_FLOAT:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<float>>(resultSetSize);
                    break;
                case COLUMN_DOUBLE:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<double>>(resultSetSize);
                    break;
                case COLUMN_POINT:
                    throw std::runtime_error("ORDER BY operation not implemented for points");
                case COLUMN_POLYGON:
                    throw std::runtime_error("ORDER BY operation not implemented for polygons");
                case COLUMN_STRING:
                    throw std::runtime_error("ORDER BY operation not implemented for strings");
                case COLUMN_INT8_T:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<int8_t>>(resultSetSize);
                    break;
                default:
                    break;
                }

                // Alloc the null collumn and zero it
                reconstructedOrderByColumnsNullMerged_[orderColumn.first] =
                    std::make_unique<int8_t[]>(resultSetNullSize);
                for (int32_t i = 0; i < resultSetNullSize; i++)
                {
                    reconstructedOrderByColumnsNullMerged_[orderColumn.first].get()[i] = 0;
                }
            }

            // Write the results to the result map
            bool dataMerged = false;
            while (dataMerged != true)
            {
                // Merge the input arrays to the output arrays
                // Check each entry from left to right (the numbers are in inverse because of the dispatcher_)
                for (int32_t i = orderByColumns_.size() - 1; i >= 0; i--)
                {
                    // Check if all values pointed to by the counters are equal, if yes - proceed to the next column
                    bool valuesAreEqual = true;
                    bool lastValueFound = true;
                    int32_t firstNonzeroMergeCounterIdx = -1;

                    // WARNING - this works only for non empty columns
                    switch (
                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][0].get()->GetType())
                    {
                    case COLUMN_INT:
                    {
                        int32_t lastValue = 0;
                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            if (lastValueFound && merge_counters[j] < merge_limits[j])
                            {
                                lastValue =
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                firstNonzeroMergeCounterIdx = j;
                                lastValueFound = false;
                            }
                            else if (merge_counters[j] < merge_limits[j])
                            {
                                int32_t value =
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (lastValue != value)
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
                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            if (lastValueFound && merge_counters[j] < merge_limits[j])
                            {
                                lastValue =
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                firstNonzeroMergeCounterIdx = j;
                                lastValueFound = false;
                            }
                            else if (merge_counters[j] < merge_limits[j])
                            {
                                int64_t value =
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (lastValue != value)
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
                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            if (lastValueFound && merge_counters[j] < merge_limits[j])
                            {
                                lastValue =
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                firstNonzeroMergeCounterIdx = j;
                                lastValueFound = false;
                            }
                            else if (merge_counters[j] < merge_limits[j])
                            {
                                float value =
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (lastValue != value)
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
                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            if (lastValueFound && merge_counters[j] < merge_limits[j])
                            {
                                lastValue =
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                firstNonzeroMergeCounterIdx = j;
                                lastValueFound = false;
                            }
                            else if (merge_counters[j] < merge_limits[j])
                            {
                                double value =
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (lastValue != value)
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
                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            if (lastValueFound && merge_counters[j] < merge_limits[j])
                            {
                                lastValue =
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                firstNonzeroMergeCounterIdx = j;
                                lastValueFound = false;
                            }
                            else if (merge_counters[j] < merge_limits[j])
                            {
                                int8_t value =
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (lastValue != value)
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
                    if (firstNonzeroMergeCounterIdx == -1)
                    {
                        dataMerged = true;
                        break;
                    }

                    // If all values in the valid merge pointers are equal - continue to the next column
                    // If this column is the last column - insert the next value and exit the loop
                    if (valuesAreEqual && i > 0)
                    {
                        continue;
                    }
                    else if (valuesAreEqual && i == 0)
                    {
                        // Instert a tuple at first nonzero place and break
                        if (resultSetCounter < resultSetSize)
                        {
                            // The program copies the result values - based on column name
                            for (auto& retColumn : reconstructedOrderByRetColumnBlocks)
                            {
                                switch (retColumn.second[0].get()->GetType())
                                {
                                case COLUMN_INT:
                                {
                                    int32_t value =
                                        dynamic_cast<VariantArray<int32_t>*>(
                                            retColumn.second[firstNonzeroMergeCounterIdx].get())
                                            ->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_LONG:
                                {
                                    int64_t value =
                                        dynamic_cast<VariantArray<int64_t>*>(
                                            retColumn.second[firstNonzeroMergeCounterIdx].get())
                                            ->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_FLOAT:
                                {
                                    float value =
                                        dynamic_cast<VariantArray<float>*>(
                                            retColumn.second[firstNonzeroMergeCounterIdx].get())
                                            ->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_DOUBLE:
                                {
                                    double value =
                                        dynamic_cast<VariantArray<double>*>(
                                            retColumn.second[firstNonzeroMergeCounterIdx].get())
                                            ->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_POINT:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for points");
                                case COLUMN_POLYGON:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for polygons");
                                case COLUMN_STRING:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for strings");
                                case COLUMN_INT8_T:
                                {
                                    int8_t value =
                                        dynamic_cast<VariantArray<int8_t>*>(
                                            retColumn.second[firstNonzeroMergeCounterIdx].get())
                                            ->getData()[merge_counters[firstNonzeroMergeCounterIdx]];
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                default:
                                    break;
                                }

                                // Write the null columns 1
                                // 1. retrieve the null value, 2. set the null value
                                int8_t nullBit =
                                    (reconstructedOrderByRetColumnNullBlocks[retColumn.first][firstNonzeroMergeCounterIdx]
                                         .get()[merge_counters[firstNonzeroMergeCounterIdx] / (sizeof(int8_t) * 8)] >>
                                     (merge_counters[firstNonzeroMergeCounterIdx] % (sizeof(int8_t) * 8))) &
                                    1;
                                nullBit <<= (resultSetCounter % (sizeof(int8_t) * 8));
                                reconstructedOrderByColumnsNullMerged_[retColumn.first].get()[resultSetCounter / 8] |=
                                    nullBit;
                            }
                            // Add to the null collumn
                            // ReconstructedOrderByOrderColumnNullBlocks[retColumn.first].get()[resultSetCounter];


                            resultSetCounter++;
                        }
                        else
                        {
                            // result set is full - need to break the while cycle - THIS MAY BE FAULTY !!!
                            dataMerged = true;
                            break;
                        }

                        merge_counters[firstNonzeroMergeCounterIdx]++;
                        break;
                    }

                    // If values are not equal
                    // If given column is ASC - find a global minimum
                    // else if given column is DESC - find a global maximum
                    // Find global minimum or maximum depending on the column type - neeed to distinguish Between different data types

                    int32_t mergeCounterIdx = -1;

                    // WARNING - this works only for non empty columns
                    switch (
                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][0].get()->GetType())
                    {
                    case COLUMN_INT:
                    {
                        int32_t minimum = std::numeric_limits<int32_t>::max();
                        int32_t maximum = std::numeric_limits<int32_t>::lowest();

                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            // Check if we are within the merged block sizes
                            if (orderByColumns_[i].second == OrderBy::Order::ASC &&
                                merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int32_t value =
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (minimum > value)
                                {
                                    minimum = value;
                                    mergeCounterIdx = j;
                                }
                            }
                            else if (orderByColumns_[i].second == OrderBy::Order::DESC &&
                                     merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int32_t value =
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (maximum < value)
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

                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            // Check if we are within the merged block sizes
                            if (orderByColumns_[i].second == OrderBy::Order::ASC &&
                                merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int64_t value =
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (minimum > value)
                                {
                                    minimum = value;
                                    mergeCounterIdx = j;
                                }
                            }
                            else if (orderByColumns_[i].second == OrderBy::Order::DESC &&
                                     merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int64_t value =
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (maximum < value)
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

                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            // Check if we are within the merged block sizes
                            if (orderByColumns_[i].second == OrderBy::Order::ASC &&
                                merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                float value =
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (minimum > value)
                                {
                                    minimum = value;
                                    mergeCounterIdx = j;
                                }
                            }
                            else if (orderByColumns_[i].second == OrderBy::Order::DESC &&
                                     merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                float value =
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (maximum < value)
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

                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            // Check if we are within the merged block sizes
                            if (orderByColumns_[i].second == OrderBy::Order::ASC &&
                                merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                double value =
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (minimum > value)
                                {
                                    minimum = value;
                                    mergeCounterIdx = j;
                                }
                            }
                            else if (orderByColumns_[i].second == OrderBy::Order::DESC &&
                                     merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                double value =
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (maximum < value)
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

                        for (int32_t j = 0; j < merge_counters.size(); j++)
                        {
                            // Check if we are within the merged block sizes
                            if (orderByColumns_[i].second == OrderBy::Order::ASC &&
                                merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int8_t value =
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (minimum > value)
                                {
                                    minimum = value;
                                    mergeCounterIdx = j;
                                }
                            }
                            else if (orderByColumns_[i].second == OrderBy::Order::DESC &&
                                     merge_counters[j] < merge_limits[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int8_t value =
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[merge_counters[j]];
                                if (maximum < value)
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
                    if (mergeCounterIdx != -1)
                    {
                        // Insert and break
                        if (resultSetCounter < resultSetSize)
                        {
                            // The program copies the result values - based on column name
                            for (auto& retColumn : reconstructedOrderByRetColumnBlocks)
                            {
                                switch (retColumn.second[0].get()->GetType())
                                {
                                case COLUMN_INT:
                                {
                                    int32_t value = dynamic_cast<VariantArray<int32_t>*>(
                                                        retColumn.second[mergeCounterIdx].get())
                                                        ->getData()[merge_counters[mergeCounterIdx]];
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_LONG:
                                {
                                    int64_t value = dynamic_cast<VariantArray<int64_t>*>(
                                                        retColumn.second[mergeCounterIdx].get())
                                                        ->getData()[merge_counters[mergeCounterIdx]];
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_FLOAT:
                                {
                                    float value = dynamic_cast<VariantArray<float>*>(
                                                      retColumn.second[mergeCounterIdx].get())
                                                      ->getData()[merge_counters[mergeCounterIdx]];
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_DOUBLE:
                                {
                                    double value = dynamic_cast<VariantArray<double>*>(
                                                       retColumn.second[mergeCounterIdx].get())
                                                       ->getData()[merge_counters[mergeCounterIdx]];
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                case COLUMN_POINT:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for points");
                                case COLUMN_POLYGON:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for polygons");
                                case COLUMN_STRING:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for strings");
                                case COLUMN_INT8_T:
                                {
                                    int8_t value = dynamic_cast<VariantArray<int8_t>*>(
                                                       retColumn.second[mergeCounterIdx].get())
                                                       ->getData()[merge_counters[mergeCounterIdx]];
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetCounter] = value;
                                }
                                break;
                                default:
                                    break;
                                }

                                // Write the null columns 2
                                int8_t nullBit =
                                    (reconstructedOrderByRetColumnNullBlocks[retColumn.first][mergeCounterIdx]
                                         .get()[merge_counters[mergeCounterIdx] / (sizeof(int8_t) * 8)] >>
                                     (merge_counters[mergeCounterIdx] % (sizeof(int8_t) * 8))) &
                                    1;
                                nullBit <<= (resultSetCounter % (sizeof(int8_t) * 8));
                                reconstructedOrderByColumnsNullMerged_[retColumn.first].get()[resultSetCounter / 8] |=
                                    nullBit;
                            }

                            resultSetCounter++;
                        }
                        else
                        {
                            // result set is full - need to break the while cycle - THIS MAY BE FAULTY !!!
                            dataMerged = true;
                        }

                        merge_counters[mergeCounterIdx]++;
                        break;
                    }
                    else
                    {
                        // ??? I dont know what happens the program gets here
                    }
                }
            }
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::info)
                << "Order by all blocks Done in thread: " << dispatcherThreadId_ << '\n';
            // Increment counter and notify threads
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::orderByMutex_);
            GpuSqlDispatcher::IncOrderByDoneCounter();
            GpuSqlDispatcher::orderByCV_.notify_all();
        }
    }
    return 0;
}
