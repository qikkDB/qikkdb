#include "GpuSqlDispatcherVMFunctions.h"
#include <array>
#include "../ParserExceptions.h"
#include "../../PointFactory.h"
#include "../../CudaLogBoost.h"
#include "../../ComplexPolygonFactory.h"
#include "../../StringFactory.h"
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::retFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, double)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, ColmnarDB::Types::Point)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::Ret, int8_t)
END_DISPATCH_TABLE

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::lockRegisterFunction_ = &GpuSqlDispatcher::LockRegister;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::getLoadSizeFunction_ = &GpuSqlDispatcher::GetLoadSize;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::filFunction_ = &GpuSqlDispatcher::Fil;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::whereEvaluationFunction_ = &GpuSqlDispatcher::WhereEvaluation;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::jmpFunction_ = &GpuSqlDispatcher::Jmp;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::doneFunction_ = &GpuSqlDispatcher::Done;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showDatabasesFunction_ = &GpuSqlDispatcher::ShowDatabases;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showTablesFunction_ = &GpuSqlDispatcher::ShowTables;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showColumnsFunction_ = &GpuSqlDispatcher::ShowColumns;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showConstraintsFunction_ = &GpuSqlDispatcher::ShowConstraints;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showQueryColumnTypesFunction_ =
    &GpuSqlDispatcher::ShowQueryColumnTypes;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::insertIntoDoneFunction_ = &GpuSqlDispatcher::InsertIntoDone;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createDatabaseFunction_ = &GpuSqlDispatcher::CreateDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropDatabaseFunction_ = &GpuSqlDispatcher::DropDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createTableFunction_ = &GpuSqlDispatcher::CreateTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropTableFunction_ = &GpuSqlDispatcher::DropTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::alterTableFunction_ = &GpuSqlDispatcher::AlterTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::alterDatabaseFunction_ = &GpuSqlDispatcher::AlterDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createIndexFunction_ = &GpuSqlDispatcher::CreateIndex;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::isNullFunction_ =
    &GpuSqlDispatcher::NullMaskCol<NullMaskOperations::isNull>;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::isNotNullFunction_ =
    &GpuSqlDispatcher::NullMaskCol<NullMaskOperations::isNotNull>;

template <>
GpuSqlDispatcher::InstructionStatus
GpuSqlDispatcher::LoadCol<ColmnarDB::Types::ComplexPolygon>(std::string& colName)
{
    if (allocatedPointers_.find(colName + "_polyPoints") == allocatedPointers_.end() &&
        !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "Load: " << colName << " " << typeid(ColmnarDB::Types::ComplexPolygon).name() << '\n';

        std::string table;
        std::string column;

        std::tie(table, column) = SplitColumnName(colName);

        const int32_t blockCount =
            database_->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
        GpuSqlDispatcher::deviceCountLimit_ =
            std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
        if (blockIndex_ >= blockCount)
        {
            return InstructionStatus::OUT_OF_BLOCKS;
        }
        if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
        {
            isLastBlockOfDevice_ = true;
        }
        if (blockIndex_ == blockCount - 1)
        {
            isOverallLastBlock_ = true;
        }

        noLoad_ = false;

        if (loadNecessary_ == 0 || loadSize_ <= 0)
        {
            instructionPointer_ = jmpInstructionPosition_;
            return InstructionStatus::LOAD_SKIPPED;
        }

        auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());


        if (!usingJoin_)
        {
            auto block =
                dynamic_cast<BlockBase<ColmnarDB::Types::ComplexPolygon>*>(col->GetBlocksList()[blockIndex_]);
            nullmask_t* nullMaskPtr = nullptr;

            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize_);
                    nullMaskPtr = AllocateRegister<nullmask_t>(colName + NULL_SUFFIX, bitMaskCapacity);

                    if (loadOffset_ > 0)
                    {
                        int32_t offsetBitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize_ + loadOffset_);
                        int32_t maxBitMaskCapacity = NullValues::GetNullBitMaskSize(block->GetSize());

                        offsetBitMaskCapacity = std::min(offsetBitMaskCapacity, maxBitMaskCapacity);

                        std::vector<nullmask_t> maskToOffset(block->GetNullBitmask(),
                                                          block->GetNullBitmask() + offsetBitMaskCapacity);
                        ShiftNullMaskLeft(maskToOffset, loadOffset_);
                        GPUMemory::copyHostToDevice(nullMaskPtr, maskToOffset.data(), bitMaskCapacity);
                    }
                    else
                    {
                        GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            InsertComplexPolygon(database_->GetName(), colName,
                                 std::vector<ColmnarDB::Types::ComplexPolygon>(block->GetData() + loadOffset_,
                                                                               block->GetData() + loadOffset_ + loadSize_),
                                 loadSize_, false, nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<ColmnarDB::Types::ComplexPolygon> joinedPolygons;
            nullmask_t* nullMaskPtr = nullptr;

            int32_t outDataSize;
            CPUJoinReorderer::reorderByJI<ColmnarDB::Types::ComplexPolygon>(joinedPolygons, outDataSize,
                                                                            *col, blockIndex_,
                                                                            joinIndices_->at(table),
                                                                            database_->GetBlockSize());

            if (col->GetIsNullable())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize);
                    auto cacheMaskEntry =
                        Context::getInstance().getCacheForCurrentDevice().getColumn<nullmask_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_,
                        bitMaskCapacity, loadSize_, loadOffset_);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        CPUJoinReorderer::reorderNullMaskByJIPushToGPU<ColmnarDB::Types::ComplexPolygon>(
                            std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex_,
                            joinIndices_->at(table), database_->GetBlockSize());
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            InsertComplexPolygon(database_->GetName(), colName, joinedPolygons, loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LoadCol<ColmnarDB::Types::Point>(std::string& colName)
{
    if (allocatedPointers_.find(colName) == allocatedPointers_.end() && !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "Load: " << colName << " " << typeid(ColmnarDB::Types::Point).name() << '\n';

        std::string table;
        std::string column;

        std::tie(table, column) = SplitColumnName(colName);

        const int32_t blockCount =
            database_->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
        GpuSqlDispatcher::deviceCountLimit_ =
            std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
        if (blockIndex_ >= blockCount)
        {
            return InstructionStatus::OUT_OF_BLOCKS;
        }
        if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
        {
            isLastBlockOfDevice_ = true;
        }
        if (blockIndex_ == blockCount - 1)
        {
            isOverallLastBlock_ = true;
        }

        noLoad_ = false;

        if (loadNecessary_ == 0 || loadSize_ <= 0)
        {
            instructionPointer_ = jmpInstructionPosition_;
            return InstructionStatus::LOAD_SKIPPED;
        }

        auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());

        if (!usingJoin_)
        {
            auto block = dynamic_cast<BlockBase<ColmnarDB::Types::Point>*>(col->GetBlocksList()[blockIndex_]);

            std::vector<NativeGeoPoint> nativePoints;
            std::transform(block->GetData() + loadOffset_, block->GetData() + loadOffset_ + loadSize_, std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint {
                return NativeGeoPoint{point.geopoint().latitude(), point.geopoint().longitude()};
            });

            auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(
                database_->GetName(), colName, blockIndex_, nativePoints.size(), loadSize_, loadOffset_);
            if (!std::get<2>(cacheEntry))
            {
                GPUMemory::copyHostToDevice(std::get<0>(cacheEntry),
                                            reinterpret_cast<NativeGeoPoint*>(nativePoints.data()),
                                            nativePoints.size());
            }
            nullmask_t* nullMaskPtr = nullptr;
            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize_);
                    auto cacheMaskEntry =
                        Context::getInstance().getCacheForCurrentDevice().getColumn<nullmask_t>(
                        database_->GetName(), colName + NULL_SUFFIX, blockIndex_, bitMaskCapacity,
                        loadSize_, loadOffset_);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        if (loadOffset_ > 0)
                        {
                            int32_t offsetBitMaskCapacity =
                                NullValues::GetNullBitMaskSize(loadSize_ + loadOffset_);
                            int32_t maxBitMaskCapacity = NullValues::GetNullBitMaskSize(block->GetSize());

                            offsetBitMaskCapacity = std::min(offsetBitMaskCapacity, maxBitMaskCapacity);

                            std::vector<nullmask_t> maskToOffset(block->GetNullBitmask(),
                                                              block->GetNullBitmask() + offsetBitMaskCapacity);
                            ShiftNullMaskLeft(maskToOffset, loadOffset_);
                            GPUMemory::copyHostToDevice(std::get<0>(cacheMaskEntry),
                                                        maskToOffset.data(), bitMaskCapacity);
                        }
                        else
                        {
                            GPUMemory::copyHostToDevice(std::get<0>(cacheMaskEntry),
                                                        block->GetNullBitmask(), bitMaskCapacity);
                        }
                    }
                    AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            AddCachedRegister(colName, std::get<0>(cacheEntry), nativePoints.size(), nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<ColmnarDB::Types::Point> joinedPoints;
            nullmask_t* nullMaskPtr = nullptr;
            int32_t outDataSize;
            CPUJoinReorderer::reorderByJI<ColmnarDB::Types::Point>(joinedPoints, outDataSize, *col,
                                                                   blockIndex_, joinIndices_->at(table),
                                                                   database_->GetBlockSize());

            std::vector<NativeGeoPoint> nativePoints;
            std::transform(joinedPoints.data(), joinedPoints.data() + loadSize, std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint {
                return NativeGeoPoint{point.geopoint().latitude(), point.geopoint().longitude()};
            });

            auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(
                database_->GetName(), joinCacheId, blockIndex_, loadSize, loadSize_, loadOffset_);
            if (!std::get<2>(cacheEntry))
            {
                GPUMemory::copyHostToDevice(std::get<0>(cacheEntry),
                                            reinterpret_cast<NativeGeoPoint*>(nativePoints.data()),
                                            nativePoints.size());
            }

            if (col->GetIsNullable())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize);
                    auto cacheMaskEntry =
                        Context::getInstance().getCacheForCurrentDevice().getColumn<nullmask_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_,
                        bitMaskCapacity, loadSize_, loadOffset_);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        CPUJoinReorderer::reorderNullMaskByJIPushToGPU<ColmnarDB::Types::Point>(
                            std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex_,
                            joinIndices_->at(table), database_->GetBlockSize());
                    }
                    AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            AddCachedRegister(colName, std::get<0>(cacheEntry), loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return InstructionStatus::CONTINUE;
}


template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LoadCol<std::string>(std::string& colName)
{
    if (allocatedPointers_.find(colName + "_allChars") == allocatedPointers_.end() &&
        !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "Load: " << colName << " " << typeid(std::string).name() << '\n';

        std::string table;
        std::string column;

        std::tie(table, column) = SplitColumnName(colName);

        const int32_t blockCount =
            database_->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
        GpuSqlDispatcher::deviceCountLimit_ =
            std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
        if (blockIndex_ >= blockCount)
        {
            return InstructionStatus::OUT_OF_BLOCKS;
        }
        if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
        {
            isLastBlockOfDevice_ = true;
        }
        if (blockIndex_ == blockCount - 1)
        {
            isOverallLastBlock_ = true;
        }

        noLoad_ = false;

        if (loadNecessary_ == 0 || loadSize_ <= 0)
        {
            instructionPointer_ = jmpInstructionPosition_;
            return InstructionStatus::LOAD_SKIPPED;
        }

        auto col = dynamic_cast<const ColumnBase<std::string>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());

        if (!usingJoin_)
        {
            auto block = dynamic_cast<BlockBase<std::string>*>(col->GetBlocksList()[blockIndex_]);
            nullmask_t* nullMaskPtr = nullptr;
            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize_);
                    nullMaskPtr = AllocateRegister<nullmask_t>(colName + NULL_SUFFIX, bitMaskCapacity);
                    if (loadOffset_ > 0)
                    {
                        int32_t offsetBitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize_ + loadOffset_);
                        int32_t maxBitMaskCapacity = NullValues::GetNullBitMaskSize(block->GetSize());

                        offsetBitMaskCapacity = std::min(offsetBitMaskCapacity, maxBitMaskCapacity);

                        std::vector<nullmask_t> maskToOffset(block->GetNullBitmask(),
                                                          block->GetNullBitmask() + offsetBitMaskCapacity);
                        ShiftNullMaskLeft(maskToOffset, loadOffset_);
                        GPUMemory::copyHostToDevice(nullMaskPtr, maskToOffset.data(), bitMaskCapacity);
                    }
                    else
                    {
                        GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            InsertString(database_->GetName(), colName, block->GetData() + loadOffset_, loadSize_,
                         false, nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<std::string> joinedStrings;
            nullmask_t* nullMaskPtr = nullptr;

            int32_t outDataSize;
            CPUJoinReorderer::reorderByJI<std::string>(joinedStrings, outDataSize, *col, blockIndex_,
                                                       joinIndices_->at(table), database_->GetBlockSize());

            if (col->GetIsNullable())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = NullValues::GetNullBitMaskSize(loadSize);
                    auto cacheMaskEntry =
                        Context::getInstance().getCacheForCurrentDevice().getColumn<nullmask_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_,
                        bitMaskCapacity, loadSize_, loadOffset_);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        CPUJoinReorderer::reorderNullMaskByJIPushToGPU<std::string>(
                            std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex_,
                            joinIndices_->at(table), database_->GetBlockSize());
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<nullmask_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            InsertString(database_->GetName(), colName, joinedStrings.data(), loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetCol<ColmnarDB::Types::ComplexPolygon>()
{
    if (usingGroupBy_)
    {
        throw RetPolygonGroupByException();
    }
    else
    {
        auto col = arguments_.Read<std::string>();
        PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
        auto alias = arguments_.Read<std::string>();

        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<ColmnarDB::Types::ComplexPolygon>(col);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "RetPolygonCol: " << col << ", thread: " << dispatcherThreadId_ << '\n';

        std::unique_ptr<std::string[]> outData(new std::string[database_->GetBlockSize()]);
        int32_t outSize;
        int32_t nullMaskPtrSize = 0;
        std::vector<nullmask_t> nullMaskVector = {};

        if (usingOrderBy_)
        {
            if (isOverallLastBlock_)
            {
                VariantArray<std::string>* reconstructedColumn = dynamic_cast<VariantArray<std::string>*>(
                    reconstructedOrderByColumnsMerged_.at(col).get());
                outData = std::move(reconstructedColumn->getDataRef());
                outSize = reconstructedColumn->GetSize();

                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector =
                    std::vector<nullmask_t>(reconstructedOrderByColumnsNullMerged_.at(col).get(),
                                         reconstructedOrderByColumnsNullMerged_.at(col).get() + nullMaskPtrSize);
            }
            else
            {
                return InstructionStatus::CONTINUE;
            }
        }
        else
        {
            auto ACol = FindCompositeDataTypeAllocation<ColmnarDB::Types::ComplexPolygon>(col);


            if (ACol.GpuNullMaskPtr)
            {
                size_t bitMaskSize = NullValues::GetNullBitMaskSize(database_->GetBlockSize());
                std::unique_ptr<nullmask_t[]> nullMask(new nullmask_t[bitMaskSize]);
                GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize, ACol.GpuPtr,
                                                        reinterpret_cast<int8_t*>(filter_),
                                                        ACol.ElementCount, nullMask.get(),
                                                        reinterpret_cast<nullmask_t*>(ACol.GpuNullMaskPtr));
                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector = std::vector<nullmask_t>(nullMask.get(), nullMask.get() + nullMaskPtrSize);
            }
            else
            {
                GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize, ACol.GpuPtr,
                                                        reinterpret_cast<int8_t*>(filter_), ACol.ElementCount);
            }
            CudaLogBoost::getInstance(CudaLogBoost::debug) << "dataSize: " << outSize << '\n';
        }

        if (outSize > 0)
        {
            ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
            InsertIntoPayload(payload, outData, outSize, payloadType);
            MergePayloadToSelfResponse(alias, col, payload, nullMaskVector);
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetCol<ColmnarDB::Types::Point>()
{
    if (usingGroupBy_)
    {
        throw RetPointGroupByException();
    }
    else
    {
        auto colName = arguments_.Read<std::string>();
        PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
        auto alias = arguments_.Read<std::string>();

        GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<ColmnarDB::Types::Point>(colName);
        if (loadFlag != InstructionStatus::CONTINUE)
        {
            return loadFlag;
        }

        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "RetPointCol: " << colName << ", thread: " << dispatcherThreadId_ << '\n';

        std::unique_ptr<std::string[]> outData(new std::string[database_->GetBlockSize()]);
        int32_t outSize;
        int32_t nullMaskPtrSize = 0;
        std::vector<nullmask_t> nullMaskVector = {};
        // ToDo: Podmienene zapnut podla velkost buffera
        // GPUMemory::hostPin(outData.get(), database_->GetBlockSize());

        if (usingOrderBy_)
        {
            if (isOverallLastBlock_)
            {
                VariantArray<std::string>* reconstructedColumn = dynamic_cast<VariantArray<std::string>*>(
                    reconstructedOrderByColumnsMerged_.at(colName).get());
                outData = std::move(reconstructedColumn->getDataRef());
                outSize = reconstructedColumn->GetSize();

                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector =
                    std::vector<nullmask_t>(reconstructedOrderByColumnsNullMerged_.at(colName).get(),
                                         reconstructedOrderByColumnsNullMerged_.at(colName).get() + nullMaskPtrSize);
            }
            else
            {
                return InstructionStatus::CONTINUE;
            }
        }
        else
        {
            PointerAllocation ACol = allocatedPointers_.at(colName);

            if (ACol.GpuNullMaskPtr)
            {
                size_t bitMaskSize = NullValues::GetNullBitMaskSize(database_->GetBlockSize());
                std::unique_ptr<nullmask_t[]> nullMask(new nullmask_t[bitMaskSize]);
                GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
                                                         reinterpret_cast<NativeGeoPoint*>(ACol.GpuPtr),
                                                         reinterpret_cast<int8_t*>(filter_),
                                                         ACol.ElementCount, nullMask.get(),
                                                         reinterpret_cast<nullmask_t*>(ACol.GpuNullMaskPtr));
                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector = std::vector<nullmask_t>(nullMask.get(), nullMask.get() + nullMaskPtrSize);
            }
            else
            {
                GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
                                                         reinterpret_cast<NativeGeoPoint*>(ACol.GpuPtr),
                                                         reinterpret_cast<int8_t*>(filter_), ACol.ElementCount);
            }
            // GPUMemory::hostUnregister(outData.get());
            CudaLogBoost::getInstance(CudaLogBoost::debug) << "dataSize: " << outSize << '\n';
        }

        if (outSize > 0)
        {
            ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
            InsertIntoPayload(payload, outData, outSize, payloadType);
            MergePayloadToSelfResponse(alias, colName, payload, nullMaskVector);
        }
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetCol<std::string>()
{
    auto colName = arguments_.Read<std::string>();
    PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
    auto alias = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "RetStringCol: " << colName << ", thread: " << dispatcherThreadId_ << '\n';

    int32_t outSize;
    std::unique_ptr<std::string[]> outData;
    int32_t nullMaskPtrSize = 0;
    std::vector<nullmask_t> nullMaskVector = {};
    if (usingGroupBy_)
    {
        if (isOverallLastBlock_)
        {
            // Return key or value col (key if groupByColumns_ contains colName)
            auto col = FindCompositeDataTypeAllocation<std::string>(
                colName + (std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                        StringDataTypeComp(colName)) != groupByColumns_.end() ?
                               KEYS_SUFFIX :
                               ""));
            outSize = col.ElementCount;

            if (usingOrderBy_)
            {
                CudaLogBoost::getInstance(CudaLogBoost::debug) << "Reordering result block." << '\n';

                GPUMemory::GPUString reorderedColumn;
                size_t inNullColSize = NullValues::GetNullBitMaskSize(outSize);
                cuda_ptr<nullmask_t> reorderedNullColumn(inNullColSize);

                PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");
                GPUOrderBy::ReOrderStringByIdx(reorderedColumn,
                                               reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                               col.GpuPtr, outSize);
                GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                                   reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                                   reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr), outSize);

                GPUMemory::free(col.GpuPtr);
                GPUMemory::free(reinterpret_cast<int8_t*>(col.GpuNullMaskPtr));

                col.GpuPtr.stringIndices = reorderedColumn.stringIndices;
                col.GpuPtr.allChars = reorderedColumn.allChars;
                col.GpuNullMaskPtr = reinterpret_cast<std::uintptr_t>(reorderedNullColumn.release());
            }

            outData = std::unique_ptr<std::string[]>(new std::string[outSize]);
            if (col.GpuNullMaskPtr)
            {
                size_t bitMaskSize = NullValues::GetNullBitMaskSize(database_->GetBlockSize());
                std::unique_ptr<nullmask_t[]> nullMask =
                    std::unique_ptr<nullmask_t[]>(new nullmask_t[bitMaskSize]);
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, col.GpuPtr, nullptr,
                                                     col.ElementCount, nullMask.get(),
                                                     reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr));
                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector = std::vector<nullmask_t>(nullMask.get(), nullMask.get() + nullMaskPtrSize);
            }
            else
            {
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, col.GpuPtr, nullptr, col.ElementCount);
            }
        }
        else
        {
            return InstructionStatus::CONTINUE;
        }
    }
    else
    {
        if (usingOrderBy_)
        {
            if (isOverallLastBlock_)
            {
                VariantArray<std::string>* reconstructedColumn = dynamic_cast<VariantArray<std::string>*>(
                    reconstructedOrderByColumnsMerged_.at(colName).get());
                outData = std::move(reconstructedColumn->getDataRef());
                outSize = reconstructedColumn->GetSize();

                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector =
                    std::vector<nullmask_t>(reconstructedOrderByColumnsNullMerged_.at(colName).get(),
                                         reconstructedOrderByColumnsNullMerged_.at(colName).get() + nullMaskPtrSize);
          }
            else
            {
                return InstructionStatus::CONTINUE;
            }
        }
        else
        {
            auto col = FindCompositeDataTypeAllocation<std::string>(colName);
            outSize = col.ElementCount;
            outData = std::unique_ptr<std::string[]>(new std::string[outSize]);
            if (col.GpuNullMaskPtr)
            {
                size_t bitMaskSize = NullValues::GetNullBitMaskSize(database_->GetBlockSize());
                std::unique_ptr<nullmask_t[]> nullMask(new nullmask_t[bitMaskSize]);
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, col.GpuPtr,
                                                     reinterpret_cast<int8_t*>(filter_),
                                                     col.ElementCount, nullMask.get(),
                                                     reinterpret_cast<nullmask_t*>(col.GpuNullMaskPtr));
                nullMaskPtrSize = NullValues::GetNullBitMaskSize(outSize);
                nullMaskVector = std::vector<nullmask_t>(nullMask.get(), nullMask.get() + nullMaskPtrSize);
            }
            else
            {
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, col.GpuPtr,
                                                     reinterpret_cast<int8_t*>(filter_), col.ElementCount);
            }
        }
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "dataSize: " << outSize << '\n';
    }

    if (outSize > 0)
    {
        ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
        InsertIntoPayload(payload, outData, outSize, payloadType);
        MergePayloadToSelfResponse(alias, colName, payload, nullMaskVector);
    }
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetConst<std::string>()
{
    std::string cnst = arguments_.Read<std::string>();
    PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "RET: cnst" << typeid(std::string).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    GpuSqlDispatcher::InstructionStatus loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    // Compute count of copies of the const
    int64_t dataElementCount = GetBlockSize();
    if (filter_)
    {
        GPUReconstruct::Sum(dataElementCount, reinterpret_cast<int8_t*>(filter_), dataElementCount);
    }

    // Create array and merge to protobuf response
    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount, payloadType);
    MergePayloadToSelfResponse(alias, cnst, payload, {});
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetConst<ColmnarDB::Types::Point>()
{
    std::string cnst = arguments_.Read<std::string>();
    PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "RET: cnst" << typeid(ColmnarDB::Types::Point).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    GpuSqlDispatcher::InstructionStatus loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    // Compute count of copies of the const
    int64_t dataElementCount = GetBlockSize();
    if (filter_)
    {
        GPUReconstruct::Sum(dataElementCount, reinterpret_cast<int8_t*>(filter_), dataElementCount);
    }

    // Create array and merge to protobuf response
    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount, payloadType);
    MergePayloadToSelfResponse(alias, cnst, payload, {});
    return InstructionStatus::CONTINUE;
}

template <>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::RetConst<ColmnarDB::Types::ComplexPolygon>()
{
    std::string cnst = arguments_.Read<std::string>();
    PayloadType payloadType = static_cast<PayloadType>(arguments_.Read<int32_t>());
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "RET: cnst" << typeid(ColmnarDB::Types::ComplexPolygon).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    GpuSqlDispatcher::InstructionStatus loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    // Compute count of copies of the const
    int64_t dataElementCount = GetBlockSize();
    if (filter_)
    {
        GPUReconstruct::Sum(dataElementCount, reinterpret_cast<int8_t*>(filter_), dataElementCount);
    }

    // Create array and merge to protobuf response
    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount, payloadType);
    MergePayloadToSelfResponse(alias, cnst, payload, {});
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LockRegister()
{
    std::string reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Locked register: " << reg << '\n';
    registerLockList_.insert(reg);
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LoadTableBlockInfo(const std::string& tableName)
{
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "TableInfo: " << tableName << '\n';

    const int32_t blockCount = GetBlockCount();
    GpuSqlDispatcher::deviceCountLimit_ =
        std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
    if (blockIndex_ >= blockCount)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
    {
        isLastBlockOfDevice_ = true;
    }
    if (blockIndex_ == blockCount - 1)
    {
        isOverallLastBlock_ = true;
    }

    noLoad_ = false;

    return InstructionStatus::CONTINUE;
}

size_t GpuSqlDispatcher::GetBlockSize(int32_t blockIndex)
{
    if (blockIndex == -1)
    {
        blockIndex = blockIndex_;
    }

    int64_t dataElementCount = 0;
    if (LoadTableBlockInfo(loadedTableName_) != InstructionStatus::CONTINUE)
    {
        return 0;
    }
    if (usingJoin_)
    {
        dataElementCount = joinIndices_->begin()->second[blockIndex].size();
    }
    else
    {
        dataElementCount =
            database_->GetTables().at(loadedTableName_).GetColumns().begin()->second->GetBlockSizeForIndex(blockIndex);
    }

    return dataElementCount;
}

int32_t GpuSqlDispatcher::GetBlockCount()
{
    return usingJoin_ ?
               joinIndices_->at(loadedTableName_).size() :
               database_->GetTables().at(loadedTableName_).GetColumns().begin()->second.get()->GetBlockCount();
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::GetLoadSize()
{
    int64_t offset = arguments_.Read<int64_t>();
    int64_t limit = arguments_.Read<int64_t>();

    bool usingWhere = arguments_.Read<bool>();
    bool usingGroupBy = arguments_.Read<bool>();
    bool usingOrderBy = arguments_.Read<bool>();
    bool usingAggregation = arguments_.Read<bool>();
    bool usingJoin = arguments_.Read<bool>();
    bool usingLoad = arguments_.Read<bool>();

    if (usingWhere || usingGroupBy || usingOrderBy || usingAggregation || usingJoin || !usingLoad)
    {
        loadOffset_ = 0;
        loadSize_ = GetBlockSize();
    }

    else
    {
        CudaLogBoost::getInstance(CudaLogBoost::info)
            << "GetLoadSize Offset: " << offset << " Limit: " << limit << '\n';

        int64_t offsetBlockIdx = 0;
        int64_t remainingOffset = offset;
        loadOffset_ = 0;
        loadSize_ = 0;

        while (offsetBlockIdx < GetBlockCount() && remainingOffset >= 0)
        {
            remainingOffset -= GetBlockSize(offsetBlockIdx++);
        }
        offsetBlockIdx--;

        int64_t offsetLimitBlockIdx = 0;
        int64_t remainingLimitOffset = offset + limit;

        while (offsetLimitBlockIdx < GetBlockCount() && remainingLimitOffset >= 0)
        {
            remainingLimitOffset -= GetBlockSize(offsetLimitBlockIdx++);
        }
        offsetLimitBlockIdx--;

        if (blockIndex_ < offsetBlockIdx || blockIndex_ > offsetLimitBlockIdx)
        {
            loadSize_ = 0;
        }

        const int64_t currentBlockSize = static_cast<int64_t>(GetBlockSize());

        if (blockIndex_ == offsetBlockIdx)
        {
            int64_t offsetBlockDataSize = 0;
            for (int32_t i = 0; i < offsetBlockIdx + 1; i++)
            {
                offsetBlockDataSize += GetBlockSize(i);
            }

            loadSize_ = std::min(offsetBlockDataSize - offset, currentBlockSize);
            loadOffset_ = std::min(offsetBlockDataSize - loadSize_, currentBlockSize);
        }

        if (blockIndex_ == offsetLimitBlockIdx)
        {
            int64_t offsetLimitBlockDataSize = 0;
            for (int32_t i = 0; i < offsetLimitBlockIdx; i++)
            {
                offsetLimitBlockDataSize += GetBlockSize(i);
            }
            loadSize_ = std::min((offset + limit) - offsetLimitBlockDataSize, currentBlockSize) - loadOffset_;
        }

        if (blockIndex_ > offsetBlockIdx && blockIndex_ < offsetLimitBlockIdx)
        {
            loadSize_ = GetBlockSize();
        }

        CudaLogBoost::getInstance(CudaLogBoost::info) << "OffsetBlockIdx: " << offsetBlockIdx << '\n';
        CudaLogBoost::getInstance(CudaLogBoost::info) << "OffsetLimitBlockIdx: " << offsetLimitBlockIdx << '\n';
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Block Load Size: " << loadSize_ << '\n';
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Block Load Offset: " << loadOffset_ << '\n';
    }

    return InstructionStatus::CONTINUE;
}

void GpuSqlDispatcher::ShiftNullMaskLeft(std::vector<nullmask_t>& mask, int64_t shift)
{
    while (shift-- > 0)
    {
        uint64_t carryBit = 0;
        for (int32_t i = mask.size() - 1; i >= 0; i--)
        {
            uint64_t newCarryBit = mask[i] & 1;
            mask[i] >>= 1;
            uint64_t a = 9223372036854775807;
            mask[i] &= a;
            mask[i] |= (carryBit << 63);
            carryBit = newCarryBit;
        }
    }
}


template <>
void GpuSqlDispatcher::FillCompositeDataTypeRegister<std::string>(GpuSqlDispatcher::CompositeDataType<std::string> column,
                                                                  const std::string& reg,
                                                                  int32_t size,
                                                                  bool useCache,
                                                                  nullmask_t* nullMaskPtr)
{
    InsertRegister(reg + "_stringIndices",
                   PointerAllocation{reinterpret_cast<uintptr_t>(column.stringIndices), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_allChars", PointerAllocation{reinterpret_cast<uintptr_t>(column.allChars), size,
                                                        !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
}

template <>
void GpuSqlDispatcher::FillCompositeDataTypeRegister<ColmnarDB::Types::ComplexPolygon>(
    GpuSqlDispatcher::CompositeDataType<ColmnarDB::Types::ComplexPolygon> column,
    const std::string& reg,
    int32_t size,
    bool useCache,
    nullmask_t* nullMaskPtr)
{
    InsertRegister(reg + "_polyPoints",
                   PointerAllocation{reinterpret_cast<uintptr_t>(column.polyPoints), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_pointIdx", PointerAllocation{reinterpret_cast<uintptr_t>(column.pointIdx), size,
                                                        !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_polyIdx", PointerAllocation{reinterpret_cast<uintptr_t>(column.polyIdx), size,
                                                       !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
}

template <>
GpuSqlDispatcher::CompositeDataType<std::string>
GpuSqlDispatcher::InsertConstCompositeDataType<std::string>(const std::string& str, size_t size)
{
    std::vector<std::string> strings(size, str);
    std::string name = "constString" + std::to_string(constStringCounter_);
    constStringCounter_++;
    return InsertString(database_->GetName(), name, strings.data(), size);
}

template <>
GpuSqlDispatcher::CompositeDataType<ColmnarDB::Types::ComplexPolygon>
GpuSqlDispatcher::InsertConstCompositeDataType<ColmnarDB::Types::ComplexPolygon>(const std::string& str, size_t size)
{
    std::string name = "constPolygon" + std::to_string(constPolygonCounter_);
    constPolygonCounter_++;
    return InsertComplexPolygon(database_->GetName(), name, {ComplexPolygonFactory::FromWkt(str)}, size);
}


GPUMemory::GPUPolygon
GpuSqlDispatcher::InsertComplexPolygon(const std::string& databaseName,
                                       const std::string& colName,
                                       const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons,
                                       int32_t size,
                                       bool useCache,
                                       nullmask_t* nullMaskPtr)
{
    if (useCache)
    {
        if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyPoints",
                                                                             blockIndex_, loadSize_, loadOffset_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointIdx",
                                                                             blockIndex_, loadSize_, loadOffset_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyIdx",
                                                                             blockIndex_, loadSize_, loadOffset_))
        {
            GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
            GPUMemory::GPUPolygon polygon;
            polygon.polyPoints =
                std::get<0>(cache.getColumn<NativeGeoPoint>(databaseName, colName + "_polyPoints",
                                                            blockIndex_, size, loadSize_, loadOffset_));
            polygon.pointIdx =
                std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_pointIdx",
                                                     blockIndex_, size, loadSize_, loadOffset_));
            polygon.polyIdx = std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_polyIdx", blockIndex_,
                                                                   size, loadSize_, loadOffset_));

            FillCompositeDataTypeRegister<ColmnarDB::Types::ComplexPolygon>(polygon, colName, size,
                                                                            useCache, nullMaskPtr);

            return polygon;
        }
        else
        {
            GPUMemory::GPUPolygon polygon =
                ComplexPolygonFactory::PrepareGPUPolygon(polygons, databaseName, colName,
                                                         blockIndex_, loadSize_, loadOffset_);
            FillCompositeDataTypeRegister<ColmnarDB::Types::ComplexPolygon>(polygon, colName, size,
                                                                            useCache, nullMaskPtr);
            return polygon;
        }
    }
    else
    {
        GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons);
        FillCompositeDataTypeRegister<ColmnarDB::Types::ComplexPolygon>(polygon, colName, size,
                                                                        useCache, nullMaskPtr);
        return polygon;
    }
}

GPUMemory::GPUString GpuSqlDispatcher::InsertString(const std::string& databaseName,
                                                    const std::string& colName,
                                                    const std::string* strings,
                                                    const size_t stringCount,
                                                    bool useCache,
                                                    nullmask_t* nullMaskPtr)
{
    if (useCache)
    {
        if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_stringIndices",
                                                                             blockIndex_, loadSize_, loadOffset_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_allChars",
                                                                             blockIndex_, loadSize_, loadOffset_))
        {
            GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
            GPUMemory::GPUString gpuString;
            gpuString.stringIndices =
                std::get<0>(cache.getColumn<int64_t>(databaseName, colName + "_stringIndices",
                                                     blockIndex_, stringCount, loadSize_, loadOffset_));
            gpuString.allChars =
                std::get<0>(cache.getColumn<char>(databaseName, colName + "_allChars", blockIndex_,
                                                  stringCount, loadSize_, loadOffset_));
            FillCompositeDataTypeRegister<std::string>(gpuString, colName, stringCount, useCache, nullMaskPtr);
            return gpuString;
        }
        else
        {
            GPUMemory::GPUString gpuString =
                StringFactory::PrepareGPUString(strings, stringCount, databaseName, colName,
                                                blockIndex_, loadSize_, loadOffset_);
            FillCompositeDataTypeRegister<std::string>(gpuString, colName, stringCount, useCache, nullMaskPtr);
            return gpuString;
        }
    }
    else
    {
        GPUMemory::GPUString gpuString = StringFactory::PrepareGPUString(strings, stringCount);
        FillCompositeDataTypeRegister<std::string>(gpuString, colName, stringCount, useCache, nullMaskPtr);
        return gpuString;
    }
}