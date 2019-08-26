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
int32_t GpuSqlDispatcher::AggregationCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();
    auto aggAsterisk = arguments_.Read<bool>();

    int32_t loadFlag = aggAsterisk ? LoadTableBlockInfo(loadedTableName_) : LoadCol<IN>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "AggCol: " << colName << " " << reg << '\n';

    PointerAllocation dummyAllocation = PointerAllocation{0, std::numeric_limits<int32_t>::max(), false, 0};
    PointerAllocation& column = aggAsterisk ? dummyAllocation : allocatedPointers_.at(colName);

    int32_t reconstructOutSize;

    IN* reconstructOutReg = nullptr;
    int8_t* reconstructOutNullMask = nullptr;
    if (std::is_same<OP, AggregationFunctions::count>::value)
    {
        if (!aggAsterisk)
        {
            // If mask is present - count suitable rows
            if (filter_)
            {
                int32_t* indexes = nullptr;
                GPUReconstruct::GenerateIndexesKeep(&indexes, &reconstructOutSize,
                                                    reinterpret_cast<int8_t*>(filter_), column.ElementCount);
                if (indexes)
                {
                    GPUMemory::free(indexes);
                }
            }
            // If mask is nullptr - count full rows
            else
            {
                reconstructOutSize = column.ElementCount;
            }
        }
        else
        {
            reconstructOutSize = GetBlockSize();
        }
    }
    else
    {
        GPUReconstruct::reconstructColKeep<IN>(&reconstructOutReg, &reconstructOutSize,
                                               reinterpret_cast<IN*>(column.GpuPtr),
                                               reinterpret_cast<int8_t*>(filter_),
                                               column.ElementCount, &reconstructOutNullMask,
                                               reinterpret_cast<int8_t*>(column.GpuNullMaskPtr));
    }

    if (column.ShouldBeFreed)
    {
        GPUMemory::free(reinterpret_cast<void*>(column.GpuPtr));
    }
    else
    {
        column.ShouldBeFreed = true;
    }

    column.GpuPtr = reinterpret_cast<uintptr_t>(reconstructOutReg);
    column.ElementCount = reconstructOutSize;
    column.GpuNullMaskPtr = reinterpret_cast<uintptr_t>(reconstructOutNullMask);

    if (!IsRegisterAllocated(reg))
    {
        // TODO: if (not COUNT operation and std::get<1>(column) == 0), set result to NaN
        OUT* result = AllocateRegister<OUT>(reg, 1);
        GPUAggregation::col<OP, OUT, IN>(result, reinterpret_cast<IN*>(column.GpuPtr), column.ElementCount);
    }
    FreeColumnIfRegister<IN>(colName);
    filter_ = 0;
    return 0;
}

template <typename OP, typename OUT, typename IN>
int32_t GpuSqlDispatcher::AggregationConst()
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "AggConst" << '\n';
    return 0;
}

template <typename OP, typename O, typename K, typename V>
class GpuSqlDispatcher::GroupByHelper
{
public:
    static std::unique_ptr<IGroupBy>
    CreateInstance(int32_t groupByBuckets, const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        return std::make_unique<GPUGroupBy<OP, O, K, V>>(Configuration::GetInstance().GetGroupByBuckets());
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        PointerAllocation groupByColumn = dispatcher.allocatedPointers_.at(groupByColumnName);

        int32_t dataSize = std::min(groupByColumn.ElementCount, valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(reinterpret_cast<K*>(groupByColumn.GpuPtr),
                           reinterpret_cast<V*>(valueColumn.GpuPtr), dataSize,
                           reinterpret_cast<int8_t*>(groupByColumn.GpuNullMaskPtr),
                           reinterpret_cast<int8_t*>(valueColumn.GpuNullMaskPtr));
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        int32_t outSize;
        K* outKeys = nullptr;
        int8_t* outKeyNullMask = nullptr;
        O* outValues = nullptr;
        int8_t* outValueNullMask = nullptr;
        reinterpret_cast<GPUGroupBy<OP, O, K, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->GetResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables_, &outKeyNullMask,
                         &outValueNullMask);
        dispatcher.InsertRegister(groupByColumnName + KEYS_SUFFIX,
                                  PointerAllocation{reinterpret_cast<uintptr_t>(outKeys), outSize, true,
                                                    reinterpret_cast<uintptr_t>(outKeyNullMask)});
        dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                         reinterpret_cast<uintptr_t>(outValueNullMask)});
    }
};

template <typename OP, typename O, typename V>
class GpuSqlDispatcher::GroupByHelper<OP, O, std::string, V>
{
public:
    static std::unique_ptr<IGroupBy>
    CreateInstance(int32_t groupByBuckets, const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        return std::make_unique<GPUGroupBy<OP, O, std::string, V>>(
            Configuration::GetInstance().GetGroupByBuckets());
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        auto groupByColumn = dispatcher.FindStringColumn(groupByColumnName);

        int32_t dataSize = std::min(std::get<1>(groupByColumn), valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(std::get<0>(groupByColumn), reinterpret_cast<V*>(valueColumn.GpuPtr),
                           dataSize, std::get<2>(groupByColumn),
                           reinterpret_cast<int8_t*>(valueColumn.GpuNullMaskPtr));
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher)
    {
        std::string groupByColumnName = groupByColumns.begin()->first;
        int32_t outSize;
        GPUMemory::GPUString outKeys;
        int8_t* outKeyNullMask = nullptr;
        O* outValues = nullptr;
        int8_t* outValueNullMask = nullptr;
        reinterpret_cast<GPUGroupBy<OP, O, std::string, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->GetResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables_, &outKeyNullMask,
                         &outValueNullMask);
        dispatcher.FillStringRegister(outKeys, groupByColumnName + KEYS_SUFFIX, outSize, true, outKeyNullMask);
        dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                         reinterpret_cast<uintptr_t>(outValueNullMask)});
    }
};

template <typename OP, typename O, typename V>
class GpuSqlDispatcher::GroupByHelper<OP, O, std::vector<void*>, V>
{
public:
    static std::unique_ptr<IGroupBy>
    CreateInstance(int32_t groupByBuckets, const std::vector<std::pair<std::string, DataType>>& groupByColumns)
    {
        std::vector<DataType> keyDataTypes;

        for (auto& groupByColumn : groupByColumns)
        {
            keyDataTypes.push_back(groupByColumn.second);
        }

        return std::make_unique<GPUGroupBy<OP, O, std::vector<void*>, V>>(
            Configuration::GetInstance().GetGroupByBuckets(), keyDataTypes);
    }

    static void ProcessBlock(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                             const PointerAllocation& valueColumn,
                             GpuSqlDispatcher& dispatcher)
    {
        std::vector<void*> keyPtrs;
        std::vector<int8_t*> keyNullMaskPtrs;
        std::vector<GPUMemory::GPUString*> stringKeyPtrs;
        int32_t minKeySize = std::numeric_limits<int32_t>::max();

        for (auto& groupByColumn : groupByColumns)
        {
            if (groupByColumn.second == DataType::COLUMN_STRING)
            {
                auto stringColumn = dispatcher.FindStringColumn(groupByColumn.first);
                GPUMemory::GPUString* stringColPtr;
                GPUMemory::alloc<GPUMemory::GPUString>(&stringColPtr, 1);

                GPUMemory::GPUString stringCol = std::get<0>(stringColumn);
                GPUMemory::copyHostToDevice<GPUMemory::GPUString>(stringColPtr, &stringCol, 1);
                keyPtrs.push_back(reinterpret_cast<void*>(stringColPtr));
                keyNullMaskPtrs.push_back(std::get<2>(stringColumn));
                stringKeyPtrs.push_back(stringColPtr);

                minKeySize = std::min(std::get<1>(stringColumn), minKeySize);
            }
            else
            {
                PointerAllocation keyColumn = dispatcher.allocatedPointers_.at(groupByColumn.first);
                keyPtrs.push_back(reinterpret_cast<void*>(keyColumn.GpuPtr));
                keyNullMaskPtrs.push_back(reinterpret_cast<int8_t*>(keyColumn.GpuNullMaskPtr));
                minKeySize = std::min(keyColumn.ElementCount, minKeySize);
            }
        }

        int32_t dataSize = std::min(minKeySize, valueColumn.ElementCount);

        reinterpret_cast<GPUGroupBy<OP, O, std::vector<void*>, V>*>(
            dispatcher.groupByTables_[dispatcher.dispatcherThreadId_].get())
            ->ProcessBlock(keyPtrs, keyNullMaskPtrs, reinterpret_cast<V*>(valueColumn.GpuPtr),
                           dataSize, reinterpret_cast<int8_t*>(valueColumn.GpuNullMaskPtr));

        for (auto& stringPtr : stringKeyPtrs)
        {
            GPUMemory::free(stringPtr);
        }
    }

    static void GetResults(const std::vector<std::pair<std::string, DataType>>& groupByColumns,
                           const std::string& reg,
                           GpuSqlDispatcher& dispatcher)
    {
        int32_t outSize;
        std::vector<void*> outKeys;
        std::vector<int8_t*> outKeysNullMasks;
        O* outValues = nullptr;
        int8_t* outValueNullMask = nullptr;
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
                dispatcher.FillStringRegister(*(reinterpret_cast<GPUMemory::GPUString*>(outKeys[i])),
                                              groupByColumns[i].first + KEYS_SUFFIX, outSize, true,
                                              outKeysNullMasks[i]);
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
        dispatcher.InsertRegister(reg, PointerAllocation{reinterpret_cast<uintptr_t>(outValues), outSize, true,
                                                         reinterpret_cast<uintptr_t>(outValueNullMask)});
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
int32_t GpuSqlDispatcher::AggregationGroupBy()
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
            int32_t loadFlag = LoadColNullMask(colTableName);
            if (loadFlag)
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
        int32_t loadFlag = LoadCol<V>(colTableName);
        if (loadFlag)
        {
            return loadFlag;
        }
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "AggGroupBy: " << colTableName << " " << reg
                                                  << ", thread: " << dispatcherThreadId_ << '\n';
    PointerAllocation& column = aggCount ? dummyAllocation : allocatedPointers_.at(colTableName);
    int32_t reconstructOutSize;

    // Reconstruct column only if it is not group by column (if it is group by column it was already reconstructed in GroupByCol)
    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colTableName)) ==
            groupByColumns_.end() &&
        !aggCount)
    {
        V* reconstructOutReg;
        int8_t* reconstructOutNullMask;
        GPUReconstruct::reconstructColKeep<V>(&reconstructOutReg, &reconstructOutSize,
                                              reinterpret_cast<V*>(column.GpuPtr),
                                              reinterpret_cast<int8_t*>(filter_),
                                              column.ElementCount, &reconstructOutNullMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr));

        if (column.ShouldBeFreed)
        {
            GPUMemory::free(reinterpret_cast<void*>(column.GpuPtr));
        }
        else
        {
            column.ShouldBeFreed = true;
        }
        column.GpuPtr = reinterpret_cast<uintptr_t>(reconstructOutReg);
        column.ElementCount = reconstructOutSize;
        column.GpuNullMaskPtr = reinterpret_cast<uintptr_t>(reconstructOutNullMask);
    }

    // TODO void param
    if (groupByTables_[dispatcherThreadId_] == nullptr)
    {
        groupByTables_[dispatcherThreadId_] = GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::CreateInstance(
            Configuration::GetInstance().GetGroupByBuckets(), groupByColumns_);
    }

    if (aggregatedRegisters_.find(reg) == aggregatedRegisters_.end())
    {
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Processed block in AggGroupBy." << '\n';
        GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::ProcessBlock(groupByColumns_, column, *this);

        // If last block was processed, reconstruct group by table
        if (isLastBlockOfDevice_)
        {
            if (isOverallLastBlock_)
            {
                // Wait until all threads finished work
                std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
                GpuSqlDispatcher::groupByCV_.wait(lock,
                                                  [] { return GpuSqlDispatcher::IsGroupByDone(); });

                CudaLogBoost::getInstance(CudaLogBoost::info)
                    << "Reconstructing group by in thread: " << dispatcherThreadId_ << '\n';

                GpuSqlDispatcher::GroupByHelper<OP, O, K, V>::GetResults(groupByColumns_, reg, *this);
            }
            else
            {
                CudaLogBoost::getInstance(CudaLogBoost::info)
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
    return 0;
}

/// This executes first (dor each block) when GROUP BY clause is used
/// It loads the group by column (if it is firt encountered reference to the column)
/// and filters it according to WHERE clause
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T>
int32_t GpuSqlDispatcher::GroupByCol()
{
    std::string columnName = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<T>(columnName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "GroupBy: " << columnName << '\n';

    PointerAllocation& column = allocatedPointers_.at(columnName);

    int32_t reconstructOutSize;
    T* reconstructOutReg;
    GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize,
                                          reinterpret_cast<T*>(column.GpuPtr),
                                          reinterpret_cast<int8_t*>(filter_), column.ElementCount);
    // TODO add null values to reconstruct

    if (column.ShouldBeFreed) // should be freed if it is not cached - if it is temp register like "YEAR(col)"
    {
        GPUMemory::free(reinterpret_cast<void*>(column.GpuPtr));
    }
    else
    {
        column.ShouldBeFreed = true; // enable future free in cleanupGpuPointers
    }

    // Now rewrite the pointer in the register (correct because the pointer is freed or stored in chache)
    column.GpuPtr = reinterpret_cast<uintptr_t>(reconstructOutReg);
    column.ElementCount = reconstructOutSize;

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(columnName)) ==
        groupByColumns_.end())
    {
        groupByColumns_.push_back({columnName, ::GetColumnType<T>()});
    }
    usingGroupBy_ = true;
    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::GroupByConst()
{
    return 0;
}
