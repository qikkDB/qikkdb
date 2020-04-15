#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/cuda_ptr.h"
#include "../../QueryEngine/GPUCore/GPUGroupBy.cuh"
#include "../../QueryEngine/GPUCore/GPUGroupByString.cuh"
#include "../../QueryEngine/GPUCore/GPUGroupByMultiKey.cuh"
#include "../../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"

/// Implementation of generic aggregation operation based on functor OP
/// Used when GROUP BY Clause is not present
/// Loads data on demand
/// COUNT operation is handled more efficiently
/// If WHERE clause is present filtering is done before agreggation
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename OUT, typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::AggregationCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();
    auto aggAsterisk = arguments_.Read<bool>();

    GpuSqlDispatcher::InstructionStatus loadFlag =
        aggAsterisk ? LoadTableBlockInfo(loadedTableName_) : LoadCol<IN>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "AggCol: " << colName << " " << reg << '\n';

    PointerAllocation dummyAllocation = PointerAllocation{0, std::numeric_limits<int32_t>::max(), false, 0};
    PointerAllocation& column = aggAsterisk ? dummyAllocation : allocatedPointers_.at(colName);

    int32_t reconstructOutSize;

    IN* reconstructOutReg = nullptr;
    int64_t* reconstructOutNullMask = nullptr;
    constexpr bool isCount = std::is_same<OP, AggregationFunctions::count>::value;
    if constexpr (isCount) // TODO consider null values
    {
        int32_t reconstructInSize = aggAsterisk ? GetBlockSize() : column.ElementCount;

        // If mask is present - count suitable rows
        if (filter_)
        {
            GPUReconstruct::Sum(reconstructOutSize, reinterpret_cast<int8_t*>(filter_), reconstructInSize);
        }
        // If mask is nullptr - count full rows
        else
        {
            reconstructOutSize = reconstructInSize;
        }
    }
    else
    {
        GPUReconstruct::reconstructColKeep<IN>(&reconstructOutReg, &reconstructOutSize,
                                               reinterpret_cast<IN*>(column.GpuPtr),
                                               reinterpret_cast<int8_t*>(filter_),
                                               column.ElementCount, &reconstructOutNullMask,
                                               reinterpret_cast<int64_t*>(column.GpuNullMaskPtr));
        // Rewrite pointers and free old ones when needed
        RewriteColumn(column, reinterpret_cast<uintptr_t>(reconstructOutReg), reconstructOutSize,
                      reconstructOutNullMask);
    }


    if (!IsRegisterAllocated(reg))
    {
        OUT* result = AllocateRegister<OUT>(reg, 1);
        if constexpr (isCount)
        {
            GPUAggregation::col<OP, OUT, IN>(result, nullptr, reconstructOutSize);
        }
        else
        {
            GPUAggregation::col<OP, OUT, IN>(result, reinterpret_cast<IN*>(column.GpuPtr), column.ElementCount);
        }
    }
    FreeColumnIfRegister<IN>(colName);
    filter_ = 0;
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename OUT, typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::AggregationConst()
{
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "AggConst" << '\n';
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename O, typename K, typename V>
class GpuSqlDispatcher::GroupByHelper
{
public:
    static std::unique_ptr<IGroupBy> CreateInstance(int32_t groupByBuckets,
                                                    int32_t hashTableMultiplier,
                                                    const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        return std::make_unique<GPUGroupBy<OP, O, K, V>>(Configuration::GetInstance().GetGroupByBuckets(),
                                                         hashTableMultiplier);
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first + RECONSTRUCTED_SUFFIX;
        PointerAllocation groupByColumn = dispatcher.allocatedPointers_.at(groupByColumnName);

        int32_t dataSize = std::min(groupByColumn.ElementCount, valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(reinterpret_cast<K*>(groupByColumn.GpuPtr),
                           reinterpret_cast<V*>(valueColumn.GpuPtr), dataSize,
                           reinterpret_cast<int64_t*>(groupByColumn.GpuNullMaskPtr),
                           reinterpret_cast<int64_t*>(valueColumn.GpuNullMaskPtr));
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher,
                           bool usingAggregation = true)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        int32_t outSize;
        K* outKeys = nullptr;
        int64_t* outKeyNullMask = nullptr;
        O* outValues = nullptr;
        int64_t* outValueNullMask = nullptr;
        reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->GetResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables_, &outKeyNullMask,
                         &outValueNullMask);
        dispatcher.InsertRegister(groupByColumnName + KEYS_SUFFIX,
                                  PointerAllocation{reinterpret_cast<uintptr_t>(outKeys), outSize, true,
                                                    reinterpret_cast<uintptr_t>(outKeyNullMask)});
        if (usingAggregation)
        {
            dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                             reinterpret_cast<uintptr_t>(outValueNullMask)});
        }
    }
};

template <typename OP, typename O, typename V>
class GpuSqlDispatcher::GroupByHelper<OP, O, std::string, V>
{
public:
    static std::unique_ptr<IGroupBy> CreateInstance(int32_t groupByBuckets,
                                                    int32_t hashTableMultiplier,
                                                    const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        return std::make_unique<GPUGroupBy<OP, O, std::string, V>>(Configuration::GetInstance().GetGroupByBuckets(),
                                                                   hashTableMultiplier);
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first + RECONSTRUCTED_SUFFIX;
        auto groupByColumn = dispatcher.FindCompositeDataTypeAllocation<std::string>(groupByColumnName);

        int32_t dataSize = std::min(groupByColumn.ElementCount, valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(groupByColumn.GpuPtr, reinterpret_cast<V*>(valueColumn.GpuPtr), dataSize,
                           reinterpret_cast<int64_t*>(groupByColumn.GpuNullMaskPtr),
                           reinterpret_cast<int64_t*>(valueColumn.GpuNullMaskPtr));
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher,
                           bool usingAggregation = true)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        int32_t outSize;
        GPUMemory::GPUString outKeys;
        int64_t* outKeyNullMask = nullptr;
        O* outValues = nullptr;
        int64_t* outValueNullMask = nullptr;
        reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->GetResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables_, &outKeyNullMask,
                         &outValueNullMask);
        dispatcher.FillCompositeDataTypeRegister<std::string>(outKeys, groupByColumnName + KEYS_SUFFIX,
                                                              outSize, true, outKeyNullMask);
        if (usingAggregation)
        {
            dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                             reinterpret_cast<uintptr_t>(outValueNullMask)});
        }
    }
};

template <typename OP, typename O, typename V>
class GpuSqlDispatcher::GroupByHelper<OP, O, std::vector<void*>, V>
{
public:
    static std::unique_ptr<IGroupBy> CreateInstance(int32_t groupByBuckets,
                                                    int32_t hashTableMultiplier,
                                                    const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        std::vector<DataType> keyDataTypes;

        for (auto& groupByColumn : groupByColumns)
        {
            keyDataTypes.push_back(groupByColumn.second);
        }

        return std::make_unique<GPUGroupBy<OP, O, std::vector<void*>, V>>(
            Configuration::GetInstance().GetGroupByBuckets(), hashTableMultiplier, keyDataTypes);
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::vector<void*> keyPtrs;
        std::vector<int64_t*> keyNullMaskPtrs;
        std::vector<GPUMemory::GPUString*> stringKeyPtrs;
        int32_t minKeySize = std::numeric_limits<int32_t>::max();

        for (auto& groupByColumn : groupByColumns)
        {
            if (groupByColumn.second == DataType::COLUMN_STRING)
            {
                std::string groupByColumnName = groupByColumn.first + RECONSTRUCTED_SUFFIX;
                auto stringColumn = dispatcher.FindCompositeDataTypeAllocation<std::string>(groupByColumnName);
                GPUMemory::GPUString* stringColPtr;
                GPUMemory::alloc<GPUMemory::GPUString>(&stringColPtr, 1);

                GPUMemory::GPUString stringCol = stringColumn.GpuPtr;
                GPUMemory::copyHostToDevice<GPUMemory::GPUString>(stringColPtr, &stringCol, 1);
                keyPtrs.push_back(reinterpret_cast<void*>(stringColPtr));
                keyNullMaskPtrs.push_back(reinterpret_cast<int64_t*>(stringColumn.GpuNullMaskPtr));
                stringKeyPtrs.push_back(stringColPtr);

                minKeySize = std::min(stringColumn.ElementCount, minKeySize);
            }
            else
            {
                std::string groupByColumnName = groupByColumn.first + RECONSTRUCTED_SUFFIX;
                PointerAllocation keyColumn = dispatcher.allocatedPointers_.at(groupByColumnName);
                keyPtrs.push_back(reinterpret_cast<void*>(keyColumn.GpuPtr));
                keyNullMaskPtrs.push_back(reinterpret_cast<int64_t*>(keyColumn.GpuNullMaskPtr));
                minKeySize = std::min(keyColumn.ElementCount, minKeySize);
            }
        }

        int32_t dataSize = std::min(minKeySize, valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, std::vector<void*>, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(keyPtrs, keyNullMaskPtrs, reinterpret_cast<V*>(valueColumn.GpuPtr),
                           dataSize, reinterpret_cast<int64_t*>(valueColumn.GpuNullMaskPtr));

        for (auto& stringPtr : stringKeyPtrs)
        {
            GPUMemory::free(stringPtr);
        }
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher,
                           bool usingAggregation = true)
    {
        int32_t outSize;
        std::vector<void*> outKeys;
        std::vector<int64_t*> outKeysNullMasks;
        O* outValues = nullptr;
        int64_t* outValueNullMask = nullptr;
        reinterpret_cast<GPUGroupBy<OP, O, std::vector<void*>, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->GetResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables_,
                         &outKeysNullMasks, &outValueNullMask);

        for (int32_t i = 0; i < groupByColumns.size(); i++)
        {
            switch (groupByColumns[i].second)
            {
            case DataType::COLUMN_INT:
                dispatcher.InsertRegister(
                    groupByColumns[i].first + KEYS_SUFFIX,
                    PointerAllocation{reinterpret_cast<uintptr_t>(reinterpret_cast<int32_t*>(outKeys[i])),
                                      outSize, true, reinterpret_cast<uintptr_t>(outKeysNullMasks[i])});
                break;
            case DataType::COLUMN_LONG:
                dispatcher.InsertRegister(
                    groupByColumns[i].first + KEYS_SUFFIX,
                    PointerAllocation{reinterpret_cast<uintptr_t>(reinterpret_cast<int64_t*>(outKeys[i])),
                                      outSize, true, reinterpret_cast<uintptr_t>(outKeysNullMasks[i])});
                break;
            case DataType::COLUMN_FLOAT:
                dispatcher.InsertRegister(
                    groupByColumns[i].first + KEYS_SUFFIX,
                    PointerAllocation{reinterpret_cast<uintptr_t>(reinterpret_cast<float*>(outKeys[i])),
                                      outSize, true, reinterpret_cast<uintptr_t>(outKeysNullMasks[i])});
                break;
            case DataType::COLUMN_DOUBLE:
                dispatcher.InsertRegister(
                    groupByColumns[i].first + KEYS_SUFFIX,
                    PointerAllocation{reinterpret_cast<uintptr_t>(reinterpret_cast<double*>(outKeys[i])),
                                      outSize, true, reinterpret_cast<uintptr_t>(outKeysNullMasks[i])});
                break;
            case DataType::COLUMN_STRING:
                dispatcher.FillCompositeDataTypeRegister<std::string>(
                    *(reinterpret_cast<GPUMemory::GPUString*>(outKeys[i])),
                    groupByColumns[i].first + KEYS_SUFFIX, outSize, true, outKeysNullMasks[i]);
                delete reinterpret_cast<GPUMemory::GPUString*>(outKeys[i]); // delete just pointer to struct
                break;
            case DataType::COLUMN_INT8_T:
                dispatcher.InsertRegister(
                    groupByColumns[i].first + KEYS_SUFFIX,
                    PointerAllocation{reinterpret_cast<uintptr_t>(reinterpret_cast<int8_t*>(outKeys[i])),
                                      outSize, true, reinterpret_cast<uintptr_t>(outKeysNullMasks[i])});
                break;
            default:
                throw std::runtime_error("GROUP BY operation does not support data type " +
                                         std::to_string(groupByColumns[i].second));
                break;
            }
        }
        if (usingAggregation)
        {
            dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                             reinterpret_cast<uintptr_t>(outValueNullMask)});
        }
    }
};

/// Implementation of generic aggregation operation based on functor OP
/// Used when GROUP BY Clause is present
/// Loads data on demand
/// If WHERE clause is present filtering is done before agreggation
/// For each block it updates group by hash table
/// To handle multi-gpu functionality - each dipatcher instance signals when it processes its last block
/// The dispatcher instance handling the overall last block waits for all other dispatcher instances to finish their last blocks
/// and saves the result of group by
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename O, typename K, typename V>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::AggregationGroupBy()
{
    auto colTableName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();
    auto aggAsterisk = arguments_.Read<bool>();

    bool aggCount = std::is_same<OP, AggregationFunctions::count>::value;

    PointerAllocation dummyAllocation;

    if (aggCount)
    {
        if (!aggAsterisk)
        {
            GpuSqlDispatcher::InstructionStatus loadFlag = LoadColNullMask(colTableName);
            if (loadFlag != InstructionStatus::CONTINUE)
            {
                return loadFlag;
            }
            auto columnMask = allocatedPointers_.at(colTableName + NULL_SUFFIX);
            int32_t columnSize = GetBlockSize();
            dummyAllocation = PointerAllocation{0, columnSize, false, columnMask.GpuPtr};
        }
        else
        {
            dummyAllocation = PointerAllocation{0, std::numeric_limits<int32_t>::max(), false, 0};
        }
        aggCount = true;
    }
    else
    {
        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<V>(colTableName);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "AggGroupBy: " << colTableName << " " << reg
                                                   << ", thread: " << dispatcherThreadId_ << '\n';
    PointerAllocation& column = aggCount ? dummyAllocation : allocatedPointers_.at(colTableName);
    int32_t reconstructOutSize;

    // Reconstruct column only if it is not group by column (if it is group by column it was already reconstructed in GroupByCol)
    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colTableName)) ==
            groupByColumns_.end() &&
        !aggCount)
    {
        V* reconstructOutReg;
        int64_t* reconstructOutNullMask;
        GPUReconstruct::reconstructColKeep<V>(&reconstructOutReg, &reconstructOutSize,
                                              reinterpret_cast<V*>(column.GpuPtr),
                                              reinterpret_cast<int8_t*>(filter_),
                                              column.ElementCount, &reconstructOutNullMask,
                                              reinterpret_cast<int64_t*>(column.GpuNullMaskPtr));
        if (reconstructOutNullMask != reinterpret_cast<int64_t*>(column.GpuNullMaskPtr))
        {
            InsertRegister(colTableName + NULL_SUFFIX + RECONSTRUCTED_SUFFIX,
                           PointerAllocation{reinterpret_cast<uintptr_t>(reconstructOutNullMask),
                                             reconstructOutSize, true, 0});
        }
        // Rewrite pointers and free old ones when needed
        RewriteColumn(column, reinterpret_cast<uintptr_t>(reconstructOutReg), reconstructOutSize,
                      reconstructOutNullMask);
    }

    if (groupByTables_[dispatcherThreadId_] == nullptr)
    {
        groupByTables_[dispatcherThreadId_] = GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::CreateInstance(
            Configuration::GetInstance().GetGroupByBuckets(), hashTableMultiplier_, groupByColumns_);
    }

    if (aggregatedRegisters_.find(reg) == aggregatedRegisters_.end())
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Processed block in AggGroupBy." << '\n';
        try
        {
            GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::ProcessBlock(groupByColumns_, column, *this);
        }
        catch (query_engine_error& err)
        {
            // If the error is not hash table full
            // or (if it is) if the hash table buffers already had max size
            if (err.GetQueryEngineError() != QueryEngineErrorType::GPU_HASH_TABLE_FULL ||
                static_cast<size_t>(Configuration::GetInstance().GetGroupByBuckets()) * hashTableMultiplier_ >= GB_BUFFER_SIZE_MAX)
            {
                throw;
            }
            // if we still can increase the hash table size, do it and restart the thread
            HandleHashTableFull();
            return InstructionStatus::CONTINUE;
        }

        // If last block was processed, reconstruct group by table
        if (isLastBlockOfDevice_)
        {
            if (isOverallLastBlock_)
            {
                // Wait until all threads finished work
                std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
                GpuSqlDispatcher::groupByCV_.wait(lock,
                                                  [] { return GpuSqlDispatcher::IsGroupByDone(); });
                if (GpuSqlDispatcher::thrownException_)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::warning)
                        << "Skip reconstruction group by in thread: " << dispatcherThreadId_ << '\n';
                    return InstructionStatus::EXCEPTION;
                }

                CudaLogBoost::getInstance(CudaLogBoost::debug)
                    << "Reconstructing group by in thread: " << dispatcherThreadId_ << '\n';

                GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::GetResults(groupByColumns_, reg, *this);
            }
            else
            {
                CudaLogBoost::getInstance(CudaLogBoost::debug)
                    << "Group by all blocks done in thread: " << dispatcherThreadId_ << '\n';
                // Increment counter and notify threads
                std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
                GpuSqlDispatcher::IncGroupByDoneCounter();
                GpuSqlDispatcher::groupByCV_.notify_all();
            }
        }
        aggregatedRegisters_.insert(reg);
    }

    FreeColumnIfRegister<V>(colTableName);
    return InstructionStatus::CONTINUE;
}

/// This executes first (dor each block) when GROUP BY clause is used
/// It loads the group by column (if it is firt encountered reference to the column)
/// and filters it according to WHERE clause
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::GroupByCol()
{
    std::string columnName = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(columnName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "GroupBy: " << columnName << '\n';

    PointerAllocation& column = allocatedPointers_.at(columnName);

    // Reconstruct key column
    int32_t reconstructOutSize;
    T* reconstructOutReg;
    int64_t* reconstructOutNullMask;
    GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize,
                                          reinterpret_cast<T*>(column.GpuPtr),
                                          reinterpret_cast<int8_t*>(filter_), column.ElementCount,
                                          &reconstructOutNullMask,
                                          reinterpret_cast<int64_t*>(column.GpuNullMaskPtr));

    InsertRegister(columnName + RECONSTRUCTED_SUFFIX,
                   PointerAllocation{reinterpret_cast<uintptr_t>(reconstructOutReg),
                                     reconstructOutSize, filter_ ? true : false,
                                     reinterpret_cast<uintptr_t>(reconstructOutNullMask)});
    InsertRegister(columnName + NULL_SUFFIX + RECONSTRUCTED_SUFFIX,
                   PointerAllocation{reinterpret_cast<uintptr_t>(reconstructOutNullMask),
                                     reconstructOutSize, filter_ ? true : false, 0});

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(columnName)) ==
        groupByColumns_.end())
    {
        groupByColumns_.push_back({columnName, ::GetColumnType<T>()});
    }
    usingGroupBy_ = true;
    return InstructionStatus::CONTINUE;
}

template <typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::GroupByConst()
{
    return InstructionStatus::CONTINUE;
}
