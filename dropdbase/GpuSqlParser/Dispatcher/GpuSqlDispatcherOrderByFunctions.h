#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/OrderByType.h"
#include "../../QueryEngine/GPUCore/cuda_ptr.h"
#include "../../VariantArray.h"

template <typename T>
int32_t GpuSqlDispatcher::OrderByCol()
{
    auto colName = arguments_.Read<std::string>();
    OrderBy::Order order = static_cast<OrderBy::Order>(arguments_.Read<int32_t>());
    int32_t columnIndex = arguments_.Read<int32_t>();

    orderByColumns_.insert({columnIndex, {colName, order}});

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    if (usingGroupBy_)
    {
        if (isOverallLastBlock_)
        {
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Order by: " << colName << '\n';
            PointerAllocation column = allocatedPointers_.at(
                colName + (std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                        StringDataTypeComp(colName)) != groupByColumns_.end() ?
                               KEYS_SUFFIX :
                               ""));
            int32_t inSize = column.ElementCount;

            if (orderByTable_ == nullptr)
            {
                orderByTable_ = std::make_unique<GPUOrderBy>(inSize);
                int32_t* orderByIndices = AllocateRegister<int32_t>("$orderByIndices", inSize);
                usingOrderBy_ = true;
            }

            PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");
            dynamic_cast<GPUOrderBy*>(orderByTable_.get())
                ->OrderByColumn(reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                reinterpret_cast<T*>(column.GpuPtr),
                                reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), inSize, order);
        }
        else
        {
            return 0;
        }
    }
    else
    {
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Order by: " << colName << '\n';
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t inSize = column.ElementCount;

        if (orderByTable_ == nullptr)
        {
            orderByTable_ = std::make_unique<GPUOrderBy>(inSize);
            int32_t* orderByIndices = AllocateRegister<int32_t>("$orderByIndices", inSize);
            usingOrderBy_ = true;
        }

        PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");
        dynamic_cast<GPUOrderBy*>(orderByTable_.get())
            ->OrderByColumn(reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                            reinterpret_cast<T*>(column.GpuPtr),
                            reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), inSize, order);
    }

    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::OrderByConst()
{
    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::OrderByReconstructCol()
{
    auto colName = arguments_.Read<std::string>();
    bool isRetColumn = arguments_.Read<bool>();

    if (!usingGroupBy_)
    {
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Reordering column: " << colName << '\n';

        int32_t loadFlag = LoadCol<T>(colName);
        if (loadFlag)
        {
            return loadFlag;
        }

        PointerAllocation col = allocatedPointers_.at(colName);
        size_t inSize = col.ElementCount;
        size_t inNullColSize = (inSize + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);

        std::unique_ptr<VariantArray<T>> outData = std::make_unique<VariantArray<T>>(inSize);
        std::unique_ptr<int8_t[]> outNullData = std::make_unique<int8_t[]>(inNullColSize);

        cuda_ptr<T> reorderedColumn(inSize);
        cuda_ptr<int8_t> reorderedNullColumn(inNullColSize);

        PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");
        GPUOrderBy::ReOrderByIdx(reorderedColumn.get(), reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                 reinterpret_cast<T*>(col.GpuPtr), col.ElementCount);
        GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                           reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                           reinterpret_cast<int8_t*>(col.GpuNullMaskPtr), inSize);

        GPUOrderBy::TransformNullValsToSmallestVal(reorderedColumn.get(), reorderedNullColumn.get(), inSize);

        int32_t outSize;
        GPUReconstruct::reconstructCol(outData->getData(), &outSize, reorderedColumn.get(),
                                       reinterpret_cast<int8_t*>(filter_), inSize,
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
    return 0;
}

template <typename T>
int32_t GpuSqlDispatcher::OrderByReconstructConst()
{
    return 0;
}