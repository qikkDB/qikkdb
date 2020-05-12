#include "GpuSqlDispatcherOrderByFunctions.h"

#include <vector>
#include <limits>
#include <cstdint>

#include "../../DataType.h"
#include "../../VariantArray.h"
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::orderByFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderBy, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderBy, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderBy, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderBy, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderBy, int8_t)
END_DISPATCH_TABLE


BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::orderByReconstructFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, double)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, ColmnarDB::Types::Point)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::OrderByReconstruct, int8_t)
END_DISPATCH_TABLE

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::freeOrderByTableFunction_ = &GpuSqlDispatcher::FreeOrderByTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::orderByReconstructRetAllBlocksFunction_ =
    &GpuSqlDispatcher::OrderByReconstructRetAllBlocks;


template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::OrderByReconstructCol<std::string>()
{
    auto colName = arguments_.Read<std::string>();
    bool isRetColumn = arguments_.Read<bool>();

    if (!usingGroupBy_)
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Reordering column: " << colName << '\n';

        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }

        auto col = FindCompositeDataTypeAllocation<std::string>(colName);
        size_t inSize = col.ElementCount;
        size_t inNullColSize = NullValues::GetNullBitMaskSize(inSize);

        std::unique_ptr<VariantArray<std::string>> outData =
            std::make_unique<VariantArray<std::string>>(inSize);
        std::unique_ptr<nullmask_t[]> outNullData(new nullmask_t[inNullColSize]);

        GPUMemory::GPUString reorderedColumn;
        cuda_ptr<nullmask_t> reorderedNullColumn(inNullColSize);
        cuda_ptr<int8_t> reorderedFilterMask(nullptr);

        PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");

        if (filter_)
        {
            reorderedFilterMask = cuda_ptr<int8_t>(inSize);
            GPUOrderBy::ReOrderByIdx(reorderedFilterMask.get(),
                                     reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                     reinterpret_cast<int8_t*>(filter_), inSize);
        }

        GPUOrderBy::ReOrderStringByIdx(reorderedColumn, reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                       col.GpuPtr, inSize);
        GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                           reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                           reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr), inSize);

        int32_t outSize;
        GPUReconstruct::ReconstructStringCol(outData->getData(), &outSize, reorderedColumn,
                                             reorderedFilterMask.get(), inSize, outNullData.get(),
                                             reorderedNullColumn.get());
        outData->resize(outSize);

        GPUMemory::free(reorderedColumn);

        if (isRetColumn)
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
        else
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::OrderByReconstructCol<ColmnarDB::Types::Point>()
{
    auto colName = arguments_.Read<std::string>();
    bool isRetColumn = arguments_.Read<bool>();

    if (!usingGroupBy_)
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Reordering column: " << colName << '\n';

        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<ColmnarDB::Types::Point>(colName);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }

        PointerAllocation col = allocatedPointers_.at(colName);
        size_t inSize = col.ElementCount;
        size_t inNullColSize = NullValues::GetNullBitMaskSize(inSize);

        std::unique_ptr<VariantArray<std::string>> outData =
            std::make_unique<VariantArray<std::string>>(inSize);
        std::unique_ptr<nullmask_t[]> outNullData(new nullmask_t[inNullColSize]);

        cuda_ptr<NativeGeoPoint> reorderedColumn(inSize);
        cuda_ptr<nullmask_t> reorderedNullColumn(inNullColSize);
        cuda_ptr<int8_t> reorderedFilterMask(nullptr);

        PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");

        if (filter_)
        {
            reorderedFilterMask = cuda_ptr<int8_t>(inSize);
            GPUOrderBy::ReOrderByIdx(reorderedFilterMask.get(),
                                     reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                     reinterpret_cast<int8_t*>(filter_), inSize);
        }

        GPUOrderBy::ReOrderByIdx(reorderedColumn.get(), reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                 reinterpret_cast<NativeGeoPoint*>(col.GpuPtr), col.ElementCount);
        GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                           reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                           reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr), inSize);

        int32_t outSize;
        GPUReconstruct::ReconstructPointColToWKT(outData->getData(), &outSize, reorderedColumn.get(),
                                                 reorderedFilterMask.get(), inSize,
                                                 outNullData.get(), reorderedNullColumn.get());
        outData->resize(outSize);

        if (isRetColumn)
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
        else
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::OrderByReconstructCol<ColmnarDB::Types::ComplexPolygon>()
{
    auto colName = arguments_.Read<std::string>();
    bool isRetColumn = arguments_.Read<bool>();

    if (!usingGroupBy_)
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Reordering column: " << colName << '\n';

        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<ColmnarDB::Types::ComplexPolygon>(colName);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }

        auto col = FindCompositeDataTypeAllocation<ColmnarDB::Types::ComplexPolygon>(colName);
        size_t inSize = col.ElementCount;
        size_t inNullColSize = NullValues::GetNullBitMaskSize(inSize);

        std::unique_ptr<VariantArray<std::string>> outData =
            std::make_unique<VariantArray<std::string>>(inSize);
        std::unique_ptr<nullmask_t[]> outNullData(new nullmask_t[inNullColSize]);

        GPUMemory::GPUPolygon reorderedColumn;
        cuda_ptr<nullmask_t> reorderedNullColumn(inNullColSize);
        cuda_ptr<int8_t> reorderedFilterMask(nullptr);

        PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");

        if (filter_)
        {
            reorderedFilterMask = cuda_ptr<int8_t>(inSize);
            GPUOrderBy::ReOrderByIdx(reorderedFilterMask.get(),
                                     reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                     reinterpret_cast<int8_t*>(filter_), inSize);
        }

        GPUOrderBy::ReOrderPolygonByIdx(reorderedColumn, reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                        col.GpuPtr, inSize);
        GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                           reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                           reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr), inSize);

        int32_t outSize;
        GPUReconstruct::ReconstructPolyColToWKT(outData->getData(), &outSize, reorderedColumn,
                                                reorderedFilterMask.get(), inSize,
                                                outNullData.get(), reorderedNullColumn.get());
        outData->resize(outSize);

        GPUMemory::free(reorderedColumn);

        if (isRetColumn)
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByRetColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
        else
        {
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnBlocks[colName].push_back(
                std::move(outData));
            orderByBlocks_[dispatcherThreadId_].ReconstructedOrderByOrderColumnNullBlocks[colName].push_back(
                std::move(outNullData));
        }
    }
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::FreeOrderByTable()
{
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Freeing order by table." << '\n';
    orderByTable_.reset();
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::OrderByReconstructRetAllBlocks()
{
    if (!usingGroupBy_ && isLastBlockOfDevice_)
    {
        if (isOverallLastBlock_)
        {
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::orderByMutex_);
            GpuSqlDispatcher::orderByCV_.wait(lock, [] { return GpuSqlDispatcher::IsOrderByDone(); });
            if (GpuSqlDispatcher::thrownException_)
            {
                CudaLogBoost::getInstance(CudaLogBoost::warning)
                    << "Skip merging partially ordered blocks in thread: " << dispatcherThreadId_ << '\n';
                return InstructionStatus::EXCEPTION;
            }

            CudaLogBoost::getInstance(CudaLogBoost::debug) << "Merging partially ordered blocks." << '\n';

            std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByOrderColumnBlocks;
            std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByRetColumnBlocks;

            std::unordered_map<std::string, std::vector<std::unique_ptr<nullmask_t[]>>> reconstructedOrderByOrderColumnNullBlocks;
            std::unordered_map<std::string, std::vector<std::unique_ptr<nullmask_t[]>>> reconstructedOrderByRetColumnNullBlocks;

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
            int32_t resultSetIdx = 0;

            // Allocate a vector of merge pointers to the input vectors - counters that hold the
            // merge positions - initialize them to zero Allocate a vector that holds the sizes of
            // the input blocks - the length of this vector equals to the number of input blocks
            int32_t blockCount = reconstructedOrderByOrderColumnBlocks.begin()->second.size();
            std::vector<int32_t> currentIndicesInBlocks(blockCount, 0);
            std::vector<int32_t> sizeOfBlocks(blockCount);

            for (int32_t i = 0; i < blockCount; i++)
            {
                int32_t blockSize = reconstructedOrderByOrderColumnBlocks.begin()->second[i]->GetSize();
                resultSetSize += blockSize;
                sizeOfBlocks[i] = blockSize;
            }

            resultSetNullSize = NullValues::GetNullBitMaskSize(resultSetSize);

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
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<std::string>>(resultSetSize);
                    break;
                case COLUMN_INT8_T:
                    reconstructedOrderByColumnsMerged_[orderColumn.first] =
                        std::make_unique<VariantArray<int8_t>>(resultSetSize);
                    break;
                default:
                    break;
                }

                // Alloc the null collumn and zero it
                reconstructedOrderByColumnsNullMerged_[orderColumn.first] =
                    std::make_unique<int64_t[]>(resultSetNullSize);
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
                    int32_t firstNonzeroBlockIdx = -1;

                    for (int32_t j = 0; j < blockCount; j++)
                    {
                        // If current block has no more elements, skip
                        if (currentIndicesInBlocks[j] == sizeOfBlocks[j])
                        {
                            continue;
                        }

                        if (firstNonzeroBlockIdx == -1)
                        {
                            firstNonzeroBlockIdx = j;
                        }
                        int32_t afterJ = j + 1;
                        // Find next block with elements
                        while (afterJ < blockCount && currentIndicesInBlocks[afterJ] == sizeOfBlocks[afterJ])
                        {
                            ++afterJ;
                        }

                        // No more blocks
                        if (afterJ == blockCount)
                        {
                            break;
                        }
                        // WARNING - this works only for non empty columns
                        // Find type and compare
                        switch (reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][0]->GetType())
                        {
                        case COLUMN_INT:
                        {
                            int32_t lastValue =
                                dynamic_cast<VariantArray<int32_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[j]];
                            int32_t value =
                                dynamic_cast<VariantArray<int32_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][afterJ]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[afterJ]];
                            if (lastValue != value)
                            {
                                valuesAreEqual = false;
                            }
                        }
                        break;
                        case COLUMN_LONG:
                        {
                            int64_t lastValue =
                                dynamic_cast<VariantArray<int64_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[j]];
                            int64_t value =
                                dynamic_cast<VariantArray<int64_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][afterJ]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[afterJ]];
                            if (lastValue != value)
                            {
                                valuesAreEqual = false;
                            }
                        }
                        break;
                        case COLUMN_FLOAT:
                        {
                            float lastValue =
                                dynamic_cast<VariantArray<float>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[j]];
                            float value =
                                dynamic_cast<VariantArray<float>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][afterJ]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[afterJ]];
                            if (lastValue != value)
                            {
                                valuesAreEqual = false;
                            }
                        }
                        break;
                        case COLUMN_DOUBLE:
                        {
                            double lastValue =
                                dynamic_cast<VariantArray<double>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[j]];
                            double value =
                                dynamic_cast<VariantArray<double>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][afterJ]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[afterJ]];
                            if (lastValue != value)
                            {
                                valuesAreEqual = false;
                            }
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
                            int8_t lastValue =
                                dynamic_cast<VariantArray<int8_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[j]];
                            int8_t value =
                                dynamic_cast<VariantArray<int8_t>*>(
                                    reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][afterJ]
                                        .get())
                                    ->getData()[currentIndicesInBlocks[afterJ]];
                            if (lastValue != value)
                            {
                                valuesAreEqual = false;
                            }
                        }
                        break;
                        default:
                            break;
                        }
                        if (!valuesAreEqual)
                        {
                            break;
                        }
                    }

                    // If no first nonzero index was found - there are no entries left - terminate the while loop
                    if (firstNonzeroBlockIdx == -1)
                    {
                        dataMerged = true;
                        break;
                    }

                    // If all values in the valid merge pointers are equal - continue to the next column
                    if (valuesAreEqual && i > 0)
                    {
                        continue;
                    }
                    // If this column is the last column - insert the next value and exit the loop
                    else if (valuesAreEqual && i == 0)
                    {
                        // Instert a tuple at first nonzero place and break
                        if (resultSetIdx < resultSetSize)
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
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_LONG:
                                {
                                    int64_t value =
                                        dynamic_cast<VariantArray<int64_t>*>(
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_FLOAT:
                                {
                                    float value =
                                        dynamic_cast<VariantArray<float>*>(
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_DOUBLE:
                                {
                                    double value =
                                        dynamic_cast<VariantArray<double>*>(
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_POINT:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for points");
                                case COLUMN_POLYGON:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for polygons");
                                case COLUMN_STRING:
                                {
                                    std::string value =
                                        dynamic_cast<VariantArray<std::string>*>(
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<std::string>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_INT8_T:
                                {
                                    int8_t value =
                                        dynamic_cast<VariantArray<int8_t>*>(
                                            retColumn.second[firstNonzeroBlockIdx].get())
                                            ->getData()[currentIndicesInBlocks[firstNonzeroBlockIdx]];
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                default:
                                    break;
                                }

                                // Write the null columns 1
                                // 1. retrieve the null value, 2. set the null value
                                int8_t nullBit = NullValues::GetConcreteBitFromBitmask(
                                    reconstructedOrderByRetColumnNullBlocks[retColumn.first][firstNonzeroBlockIdx]
                                        .get(),
                                    currentIndicesInBlocks[firstNonzeroBlockIdx]);
                                nullBit <<= NullValues::GetShiftMaskIdx(resultSetIdx);
                                reconstructedOrderByColumnsNullMerged_[retColumn.first][resultSetIdx / (sizeof(nullmask_t) * 8)] |=
                                    nullBit;
                            }
                            // Add to the null collumn
                            // ReconstructedOrderByOrderColumnNullBlocks[retColumn.first].get()[resultSetIdx];


                            resultSetIdx++;
                        }
                        else
                        {
                            throw std::out_of_range(
                                "MergeSort attempt to insert result to full dataset");
                        }

                        currentIndicesInBlocks[firstNonzeroBlockIdx]++;
                        break;
                    }

                    // If values are not equal
                    // If given column is ASC - find a global minimum
                    // else if given column is DESC - find a global maximum
                    // Find global minimum or maximum depending on the column type - neeed to distinguish Between different data types

                    int32_t blockToMergeIdx = -1;

                    // WARNING - this works only for non empty columns
                    switch (reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][0]->GetType())
                    {
                    case COLUMN_INT:
                    {
                        int32_t minimum = std::numeric_limits<int32_t>::max();
                        int32_t maximum = std::numeric_limits<int32_t>::lowest();

                        for (int32_t j = 0; j < blockCount; j++)
                        {
                            if (currentIndicesInBlocks[j] < sizeOfBlocks[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int32_t value =
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[currentIndicesInBlocks[j]];
                                if (orderByColumns_[i].second == OrderBy::Order::ASC)
                                {
                                    if (minimum > value)
                                    {
                                        minimum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                                else
                                {
                                    if (maximum < value)
                                    {
                                        maximum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                            }
                        }
                    }
                    break;
                    case COLUMN_LONG:
                    {
                        int64_t minimum = std::numeric_limits<int64_t>::max();
                        int64_t maximum = std::numeric_limits<int64_t>::lowest();

                        for (int32_t j = 0; j < blockCount; j++)
                        {
                            if (currentIndicesInBlocks[j] < sizeOfBlocks[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int64_t value =
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[currentIndicesInBlocks[j]];
                                if (orderByColumns_[i].second == OrderBy::Order::ASC)
                                {
                                    if (minimum > value)
                                    {
                                        minimum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                                else
                                {
                                    if (maximum < value)
                                    {
                                        maximum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                            }
                        }
                    }
                    break;
                    case COLUMN_FLOAT:
                    {
                        float minimum = std::numeric_limits<float>::max();
                        float maximum = std::numeric_limits<float>::lowest();

                        for (int32_t j = 0; j < blockCount; j++)
                        {
                            if (currentIndicesInBlocks[j] < sizeOfBlocks[j])
                            {
                                // Get the value from the block to which the merge counter points
                                float value =
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[currentIndicesInBlocks[j]];
                                if (orderByColumns_[i].second == OrderBy::Order::ASC)
                                {
                                    if (minimum > value)
                                    {
                                        minimum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                                else
                                {
                                    if (maximum < value)
                                    {
                                        maximum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                            }
                        }
                    }
                    break;
                    case COLUMN_DOUBLE:
                    {
                        double minimum = std::numeric_limits<double>::max();
                        double maximum = std::numeric_limits<double>::lowest();

                        for (int32_t j = 0; j < blockCount; j++)
                        {
                            if (currentIndicesInBlocks[j] < sizeOfBlocks[j])
                            {
                                // Get the value from the block to which the merge counter points
                                double value =
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[currentIndicesInBlocks[j]];
                                if (orderByColumns_[i].second == OrderBy::Order::ASC)
                                {
                                    if (minimum > value)
                                    {
                                        minimum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                                else
                                {
                                    if (maximum < value)
                                    {
                                        maximum = value;
                                        blockToMergeIdx = j;
                                    }
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

                        for (int32_t j = 0; j < blockCount; j++)
                        {
                            if (currentIndicesInBlocks[j] < sizeOfBlocks[j])
                            {
                                // Get the value from the block to which the merge counter points
                                int8_t value =
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByOrderColumnBlocks[orderByColumns_[i].first][j]
                                            .get())
                                        ->getData()[currentIndicesInBlocks[j]];
                                if (orderByColumns_[i].second == OrderBy::Order::ASC)
                                {
                                    if (minimum > value)
                                    {
                                        minimum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                                else
                                {
                                    if (maximum < value)
                                    {
                                        maximum = value;
                                        blockToMergeIdx = j;
                                    }
                                }
                            }
                        }
                    }
                    break;
                    default:
                        break;
                    }

                    // If an extrem was found (min or max)
                    if (blockToMergeIdx != -1)
                    {
                        // Insert and break
                        if (resultSetIdx < resultSetSize)
                        {
                            // The program copies the result values - based on column name
                            for (auto& retColumn : reconstructedOrderByRetColumnBlocks)
                            {
                                switch (retColumn.second[blockToMergeIdx]->GetType())
                                {
                                case COLUMN_INT:
                                {
                                    int32_t value = dynamic_cast<VariantArray<int32_t>*>(
                                                        retColumn.second[blockToMergeIdx].get())
                                                        ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<int32_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_LONG:
                                {
                                    int64_t value = dynamic_cast<VariantArray<int64_t>*>(
                                                        retColumn.second[blockToMergeIdx].get())
                                                        ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<int64_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_FLOAT:
                                {
                                    float value = dynamic_cast<VariantArray<float>*>(
                                                      retColumn.second[blockToMergeIdx].get())
                                                      ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<float>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_DOUBLE:
                                {
                                    double value = dynamic_cast<VariantArray<double>*>(
                                                       retColumn.second[blockToMergeIdx].get())
                                                       ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<double>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_POINT:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for points");
                                case COLUMN_POLYGON:
                                    throw std::runtime_error(
                                        "ORDER BY operation not implemented for polygons");
                                case COLUMN_STRING:
                                {
                                    std::string value =
                                        dynamic_cast<VariantArray<std::string>*>(
                                            retColumn.second[blockToMergeIdx].get())
                                            ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<std::string>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                case COLUMN_INT8_T:
                                {
                                    int8_t value = dynamic_cast<VariantArray<int8_t>*>(
                                                       retColumn.second[blockToMergeIdx].get())
                                                       ->getData()[currentIndicesInBlocks[blockToMergeIdx]];
                                    dynamic_cast<VariantArray<int8_t>*>(
                                        reconstructedOrderByColumnsMerged_[retColumn.first].get())
                                        ->getData()[resultSetIdx] = value;
                                }
                                break;
                                default:
                                    break;
                                }
                                // Write the null columns 2
                                int8_t nullBit = NullValues::GetConcreteBitFromBitmask(
                                    reconstructedOrderByRetColumnNullBlocks[retColumn.first][blockToMergeIdx]
                                        .get(),
                                    currentIndicesInBlocks[blockToMergeIdx]);
                                nullBit <<= NullValues::GetShiftMaskIdx(resultSetIdx);
                                reconstructedOrderByColumnsNullMerged_[retColumn.first][resultSetIdx / (sizeof(nullmask_t) * 8)] |= nullBit;
                            }

                            resultSetIdx++;
                        }
                        else
                        {
                            throw std::out_of_range(
                                "MergeSort attempt to insert result to full dataset");
                        }

                        currentIndicesInBlocks[blockToMergeIdx]++;
                        break;
                    }
                    else
                    {
                        throw std::out_of_range(
                            "MergeSort all blocks empty when looking for an extreme");
                    }
                }
            }
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug)
                << "Order by all blocks Done in thread: " << dispatcherThreadId_ << '\n';
            // Increment counter and notify threads
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::orderByMutex_);
            GpuSqlDispatcher::IncOrderByDoneCounter();
            GpuSqlDispatcher::orderByCV_.notify_all();
        }
    }
    return InstructionStatus::CONTINUE;
}
