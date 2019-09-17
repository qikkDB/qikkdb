#include "GpuSqlDispatcherVMFunctions.h"
#include <array>
#include "../ParserExceptions.h"
#include "../../PointFactory.h"
#include "../../CudaLogBoost.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::retFunctions_ = {
    &GpuSqlDispatcher::RetConst<int32_t>,
    &GpuSqlDispatcher::RetConst<int64_t>,
    &GpuSqlDispatcher::RetConst<float>,
    &GpuSqlDispatcher::RetConst<double>,
    &GpuSqlDispatcher::RetConst<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::RetConst<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::RetConst<std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int8_t>,
    &GpuSqlDispatcher::RetCol<int32_t>,
    &GpuSqlDispatcher::RetCol<int64_t>,
    &GpuSqlDispatcher::RetCol<float>,
    &GpuSqlDispatcher::RetCol<double>,
    &GpuSqlDispatcher::RetCol<ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::RetCol<ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::RetCol<std::string>,
    &GpuSqlDispatcher::RetCol<int8_t>};
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::lockRegisterFunction_ = &GpuSqlDispatcher::LockRegister;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::filFunction_ = &GpuSqlDispatcher::Fil;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::whereEvaluationFunction_ = &GpuSqlDispatcher::WhereEvaluation;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::jmpFunction_ = &GpuSqlDispatcher::Jmp;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::doneFunction_ = &GpuSqlDispatcher::Done;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showDatabasesFunction_ = &GpuSqlDispatcher::ShowDatabases;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showTablesFunction_ = &GpuSqlDispatcher::ShowTables;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showColumnsFunction_ = &GpuSqlDispatcher::ShowColumns;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::insertIntoDoneFunction_ = &GpuSqlDispatcher::InsertIntoDone;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createDatabaseFunction_ = &GpuSqlDispatcher::CreateDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropDatabaseFunction_ = &GpuSqlDispatcher::DropDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createTableFunction_ = &GpuSqlDispatcher::CreateTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropTableFunction_ = &GpuSqlDispatcher::DropTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::alterTableFunction_ = &GpuSqlDispatcher::AlterTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createIndexFunction_ = &GpuSqlDispatcher::CreateIndex;

template <>
int32_t GpuSqlDispatcher::LoadCol<ColmnarDB::Types::ComplexPolygon>(std::string& colName)
{
    if (allocatedPointers_.find(colName + "_polyPoints") == allocatedPointers_.end() &&
        !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::info)
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
            return 1;
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

        if (loadNecessary_ == 0)
        {
            instructionPointer_ = jmpInstructionPosition_;
            return 12;
        }

        auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());


        if (!usingJoin_)
        {
            auto block =
                dynamic_cast<BlockBase<ColmnarDB::Types::ComplexPolygon>*>(col->GetBlocksList()[blockIndex_]);
            int8_t* nullMaskPtr = nullptr;
            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity =
                        ((block->GetSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    nullMaskPtr = AllocateRegister<int8_t>(colName + NULL_SUFFIX, bitMaskCapacity);
                    GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            InsertComplexPolygon(database_->GetName(), colName,
                                 std::vector<ColmnarDB::Types::ComplexPolygon>(block->GetData(),
                                                                               block->GetData() +
                                                                                   block->GetSize()),
                                 block->GetSize(), false, nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<ColmnarDB::Types::ComplexPolygon> joinedPolygons;
            int8_t* nullMaskPtr = nullptr;

            int32_t outDataSize;
            GPUJoin::reorderByJoinTableCPU<ColmnarDB::Types::ComplexPolygon>(joinedPolygons, outDataSize,
                                                                             *col, blockIndex_,
                                                                             joinIndices_->at(table),
                                                                             database_->GetBlockSize());

            if (col->GetIsNullable())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_, bitMaskCapacity);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        GPUJoin::reorderNullMaskByJoinTableCPU<ColmnarDB::Types::ComplexPolygon>(
                            std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex_,
                            joinIndices_->at(table), database_->GetBlockSize());
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            InsertComplexPolygon(database_->GetName(), colName, joinedPolygons, loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return 0;
}

template <>
int32_t GpuSqlDispatcher::LoadCol<ColmnarDB::Types::Point>(std::string& colName)
{
    if (allocatedPointers_.find(colName) == allocatedPointers_.end() && !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::info)
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
            return 1;
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

        if (loadNecessary_ == 0)
        {
            instructionPointer_ = jmpInstructionPosition_;
            return 12;
        }

        auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());

        if (!usingJoin_)
        {
            auto block = dynamic_cast<BlockBase<ColmnarDB::Types::Point>*>(col->GetBlocksList()[blockIndex_]);

            std::vector<NativeGeoPoint> nativePoints;
            std::transform(block->GetData(), block->GetData() + block->GetSize(), std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint {
                return NativeGeoPoint{point.geopoint().latitude(), point.geopoint().longitude()};
            });

            auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(
                database_->GetName(), colName, blockIndex_, nativePoints.size());
            if (!std::get<2>(cacheEntry))
            {
                GPUMemory::copyHostToDevice(std::get<0>(cacheEntry),
                                            reinterpret_cast<NativeGeoPoint*>(nativePoints.data()),
                                            nativePoints.size());
            }
            int8_t* nullMaskPtr = nullptr;
            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity =
                        ((block->GetSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
                        database_->GetName(), colName + NULL_SUFFIX, blockIndex_, bitMaskCapacity);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        GPUMemory::copyHostToDevice(std::get<0>(cacheMaskEntry),
                                                    block->GetNullBitmask(), bitMaskCapacity);
                    }
                    AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            AddCachedRegister(colName, std::get<0>(cacheEntry), nativePoints.size(), nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<ColmnarDB::Types::Point> joinedPoints;
            int8_t* nullMaskPtr = nullptr;
            int32_t outDataSize;
            GPUJoin::reorderByJoinTableCPU<ColmnarDB::Types::Point>(joinedPoints, outDataSize, *col,
                                                                    blockIndex_, joinIndices_->at(table),
                                                                    database_->GetBlockSize());

            std::vector<NativeGeoPoint> nativePoints;
            std::transform(joinedPoints.data(), joinedPoints.data() + loadSize, std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint {
                return NativeGeoPoint{point.geopoint().latitude(), point.geopoint().longitude()};
            });

            auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(
                database_->GetName(), joinCacheId, blockIndex_, loadSize);
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
                    int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_, bitMaskCapacity);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        GPUJoin::reorderNullMaskByJoinTableCPU<ColmnarDB::Types::Point>(
                            std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex_,
                            joinIndices_->at(table), database_->GetBlockSize());
                    }
                    AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            AddCachedRegister(colName, std::get<0>(cacheEntry), loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return 0;
}


template <>
int32_t GpuSqlDispatcher::LoadCol<std::string>(std::string& colName)
{
    if (allocatedPointers_.find(colName + "_allChars") == allocatedPointers_.end() &&
        !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::info)
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
            return 1;
        }
        if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
        {
            isLastBlockOfDevice_ = true;
        }
        if (blockIndex_ == blockCount - 1)
        {
            isOverallLastBlock_ = true;
        }

        auto col = dynamic_cast<const ColumnBase<std::string>*>(
            database_->GetTables().at(table).GetColumns().at(column).get());

        if (!usingJoin_)
        {
            auto block = dynamic_cast<BlockBase<std::string>*>(col->GetBlocksList()[blockIndex_]);
            int8_t* nullMaskPtr = nullptr;
            if (block->GetNullBitmask())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity =
                        ((block->GetSize() + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    nullMaskPtr = AllocateRegister<int8_t>(colName + NULL_SUFFIX, bitMaskCapacity);
                    GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }
            InsertString(database_->GetName(), colName,
                         std::vector<std::string>(block->GetData(), block->GetData() + block->GetSize()),
                         block->GetSize(), false, nullMaskPtr);
            noLoad_ = false;
        }
        else
        {
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Loading joined block." << '\n';
            int32_t loadSize = joinIndices_->at(table)[blockIndex_].size();
            std::string joinCacheId = colName + "_join";
            for (auto& joinTable : *joinIndices_)
            {
                joinCacheId += "_" + joinTable.first;
            }

            std::vector<std::string> joinedStrings;
            int8_t* nullMaskPtr = nullptr;

            int32_t outDataSize;
            GPUJoin::reorderByJoinTableCPU<std::string>(joinedStrings, outDataSize, *col, blockIndex_,
                                                        joinIndices_->at(table), database_->GetBlockSize());

            if (col->GetIsNullable())
            {
                if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end())
                {
                    int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
                        database_->GetName(), joinCacheId + NULL_SUFFIX, blockIndex_, bitMaskCapacity);
                    nullMaskPtr = std::get<0>(cacheMaskEntry);
                    if (!std::get<2>(cacheMaskEntry))
                    {
                        int32_t outMaskSize;
                        GPUJoin::reorderNullMaskByJoinTableCPU<std::string>(std::get<0>(cacheMaskEntry),
                                                                            outMaskSize, *col, blockIndex_,
                                                                            joinIndices_->at(table),
                                                                            database_->GetBlockSize());
                    }
                }
                else
                {
                    nullMaskPtr =
                        reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + NULL_SUFFIX).GpuPtr);
                }
            }

            InsertString(database_->GetName(), colName, joinedStrings, loadSize, nullMaskPtr);
            noLoad_ = false;
        }
    }
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetCol<ColmnarDB::Types::ComplexPolygon>()
{
    if (usingGroupBy_)
    {
        throw RetPolygonGroupByException();
    }
    else
    {
        auto col = arguments_.Read<std::string>();
        auto alias = arguments_.Read<std::string>();

        int32_t loadFlag = LoadCol<ColmnarDB::Types::ComplexPolygon>(col);
        if (loadFlag)
        {
            return loadFlag;
        }
        CudaLogBoost::getInstance(CudaLogBoost::info)
            << "RetPolygonCol: " << col << ", thread: " << dispatcherThreadId_ << '\n';

        std::unique_ptr<std::string[]> outData(new std::string[database_->GetBlockSize()]);
        int32_t outSize;
        std::string nullMaskString = "";

        if (usingOrderBy_)
        {
            if (isOverallLastBlock_)
            {
                VariantArray<std::string>* reconstructedColumn = dynamic_cast<VariantArray<std::string>*>(
                    reconstructedOrderByColumnsMerged_.at(col).get());
                outData = std::move(reconstructedColumn->getDataRef());
                outSize = reconstructedColumn->GetSize();

                size_t bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(
                                                 reconstructedOrderByColumnsNullMerged_.at(col).get()),
                                             bitMaskSize);
            }
            else
            {
                return 0;
            }
        }
        else
        {
            std::tuple<GPUMemory::GPUPolygon, int32_t, int8_t*> ACol = FindComplexPolygon(col);

            if (std::get<2>(ACol))
            {
                size_t bitMaskSize = (database_->GetBlockSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                std::unique_ptr<int8_t[]> nullMask(new int8_t[bitMaskSize]);
                GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize, std::get<0>(ACol),
                                                        reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol),
                                                        nullMask.get(), std::get<2>(ACol));
                bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
            }
            else
            {
                GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize, std::get<0>(ACol),
                                                        reinterpret_cast<int8_t*>(filter_),
                                                        std::get<1>(ACol));
            }
            CudaLogBoost::getInstance(CudaLogBoost::info) << "dataSize: " << outSize << '\n';
        }

        if (outSize > 0)
        {
            ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
            InsertIntoPayload(payload, outData, outSize);
            MergePayloadToSelfResponse(alias, payload, nullMaskString);
        }
    }
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetCol<ColmnarDB::Types::Point>()
{
    if (usingGroupBy_)
    {
        throw RetPointGroupByException();
    }
    else
    {
        auto colName = arguments_.Read<std::string>();
        auto alias = arguments_.Read<std::string>();

        int32_t loadFlag = LoadCol<ColmnarDB::Types::Point>(colName);
        if (loadFlag)
        {
            return loadFlag;
        }

        CudaLogBoost::getInstance(CudaLogBoost::info)
            << "RetPointCol: " << colName << ", thread: " << dispatcherThreadId_ << '\n';

        std::unique_ptr<std::string[]> outData(new std::string[database_->GetBlockSize()]);
        int32_t outSize;
        std::string nullMaskString = "";
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

                size_t bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(
                                                 reconstructedOrderByColumnsNullMerged_.at(colName).get()),
                                             bitMaskSize);
            }
            else
            {
                return 0;
            }
        }
        else
        {
            PointerAllocation ACol = allocatedPointers_.at(colName);

            if (ACol.GpuNullMaskPtr)
            {
                size_t bitMaskSize = (database_->GetBlockSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                std::unique_ptr<int8_t[]> nullMask(new int8_t[bitMaskSize]);
                GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
                                                         reinterpret_cast<NativeGeoPoint*>(ACol.GpuPtr),
                                                         reinterpret_cast<int8_t*>(filter_),
                                                         ACol.ElementCount, nullMask.get(),
                                                         reinterpret_cast<int8_t*>(ACol.GpuNullMaskPtr));
                bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
            }
            else
            {
                GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
                                                         reinterpret_cast<NativeGeoPoint*>(ACol.GpuPtr),
                                                         reinterpret_cast<int8_t*>(filter_), ACol.ElementCount);
            }
            // GPUMemory::hostUnregister(outData.get());
            CudaLogBoost::getInstance(CudaLogBoost::info) << "dataSize: " << outSize << '\n';
        }

        if (outSize > 0)
        {
            ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
            InsertIntoPayload(payload, outData, outSize);
            MergePayloadToSelfResponse(alias, payload, nullMaskString);
        }
    }
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetCol<std::string>()
{
    auto colName = arguments_.Read<std::string>();
    auto alias = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "RetStringCol: " << colName << ", thread: " << dispatcherThreadId_ << '\n';

    int32_t outSize;
    std::unique_ptr<std::string[]> outData;
    std::string nullMaskString = "";
    if (usingGroupBy_)
    {
        if (isOverallLastBlock_)
        {
            // Return key or value col (key if groupByColumns_ contains colName)
            auto col = FindStringColumn(
                colName + (std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                        StringDataTypeComp(colName)) != groupByColumns_.end() ?
                               KEYS_SUFFIX :
                               ""));
            outSize = std::get<1>(col);

            if (usingOrderBy_)
            {
                CudaLogBoost::getInstance(CudaLogBoost::info) << "Reordering result block." << '\n';

                GPUMemory::GPUString reorderedColumn;
                size_t inNullColSize = (outSize + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
                cuda_ptr<int8_t> reorderedNullColumn(inNullColSize);

                PointerAllocation orderByIndices = allocatedPointers_.at("$orderByIndices");
                GPUOrderBy::ReOrderStringByIdx(reorderedColumn,
                                               reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                               std::get<0>(col), outSize);
                GPUOrderBy::ReOrderNullValuesByIdx(reorderedNullColumn.get(),
                                                   reinterpret_cast<int32_t*>(orderByIndices.GpuPtr),
                                                   std::get<2>(col), outSize);

                GPUMemory::free(std::get<0>(col));
                GPUMemory::free(std::get<2>(col));

                std::get<0>(col).stringIndices = reorderedColumn.stringIndices;
                std::get<0>(col).allChars = reorderedColumn.allChars;
                std::get<2>(col) = reorderedNullColumn.release();
            }

            outData = std::unique_ptr<std::string[]>(new std::string[outSize]);
            if (std::get<2>(col))
            {
                size_t bitMaskSize = (database_->GetBlockSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, std::get<0>(col), nullptr,
                                                     std::get<1>(col), nullMask.get(), std::get<2>(col));
                bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
            }
            else
            {
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, std::get<0>(col),
                                                     nullptr, std::get<1>(col));
            }
        }
        else
        {
            return 0;
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

                size_t bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(
                                                 reconstructedOrderByColumnsNullMerged_.at(colName).get()),
                                             bitMaskSize);
            }
            else
            {
                return 0;
            }
        }
        else
        {
            auto col = FindStringColumn(colName);
            outSize = std::get<1>(col);
            outData = std::unique_ptr<std::string[]>(new std::string[outSize]);
            if (std::get<2>(col))
            {
                size_t bitMaskSize = (database_->GetBlockSize() + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                std::unique_ptr<int8_t[]> nullMask(new int8_t[bitMaskSize]);
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, std::get<0>(col),
                                                     reinterpret_cast<int8_t*>(filter_),
                                                     std::get<1>(col), nullMask.get(), std::get<2>(col));
                bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
            }
            else
            {
                GPUReconstruct::ReconstructStringCol(outData.get(), &outSize, std::get<0>(col),
                                                     reinterpret_cast<int8_t*>(filter_), std::get<1>(col));
            }
        }
        CudaLogBoost::getInstance(CudaLogBoost::info) << "dataSize: " << outSize << '\n';
    }

    if (outSize > 0)
    {
        ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
        InsertIntoPayload(payload, outData, outSize);
        MergePayloadToSelfResponse(alias, payload, nullMaskString);
    }
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetConst<std::string>()
{
    std::string cnst = arguments_.Read<std::string>();
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info) << "RET: cnst" << typeid(std::string).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    int32_t loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag)
    {
        return loadFlag;
    }

    int64_t dataElementCount = GetBlockSize();

    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount);
    MergePayloadToSelfResponse(alias, payload, "");
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetConst<ColmnarDB::Types::Point>()
{
    std::string cnst = arguments_.Read<std::string>();
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "RET: cnst" << typeid(ColmnarDB::Types::Point).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    int32_t loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag)
    {
        return loadFlag;
    }

    int64_t dataElementCount = GetBlockSize();

    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount);
    MergePayloadToSelfResponse(alias, payload, "");
    return 0;
}

template <>
int32_t GpuSqlDispatcher::RetConst<ColmnarDB::Types::ComplexPolygon>()
{
    std::string cnst = arguments_.Read<std::string>();
    std::string alias = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "RET: cnst" << typeid(ColmnarDB::Types::ComplexPolygon).name() << '\n';

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    int32_t loadFlag = LoadTableBlockInfo(loadedTableName_);
    if (loadFlag)
    {
        return loadFlag;
    }

    int64_t dataElementCount = GetBlockSize();

    std::unique_ptr<std::string[]> outData(new std::string[dataElementCount]);
    std::fill(outData.get(), outData.get() + dataElementCount, cnst);
    InsertIntoPayload(payload, outData, dataElementCount);
    MergePayloadToSelfResponse(alias, payload, "");
    return 0;
}

int32_t GpuSqlDispatcher::LockRegister()
{
    std::string reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Locked register: " << reg << '\n';
    registerLockList_.insert(reg);
    return 0;
}

int32_t GpuSqlDispatcher::LoadTableBlockInfo(const std::string& tableName)
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "TableInfo: " << tableName << '\n';

    const int32_t blockCount =
        usingJoin_ ?
            joinIndices_->at(tableName).size() :
            database_->GetTables().at(tableName).GetColumns().begin()->second.get()->GetBlockCount();
    GpuSqlDispatcher::deviceCountLimit_ =
        std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
    if (blockIndex_ >= blockCount)
    {
        return 1;
    }
    if (blockIndex_ >= blockCount - Context::getInstance().getDeviceCount())
    {
        isLastBlockOfDevice_ = true;
    }
    if (blockIndex_ == blockCount - 1)
    {
        isOverallLastBlock_ = true;
    }

    return 0;
}

size_t GpuSqlDispatcher::GetBlockSize()
{
    int64_t dataElementCount = 0;
    if (LoadTableBlockInfo(loadedTableName_) != 0)
    {
        return 0;
    }
    if (usingJoin_)
    {
        dataElementCount = joinIndices_->begin()->second[blockIndex_].size();
    }
    else
    {
        dataElementCount =
            database_->GetTables().at(loadedTableName_).GetColumns().begin()->second->GetBlockSizeForIndex(blockIndex_);
    }
    if (filter_)
    {
        cuda_ptr<int64_t> outSum(1);
        GPUReconstruct::Sum(outSum.get(), reinterpret_cast<int8_t*>(filter_), dataElementCount);
        GPUMemory::copyDeviceToHost(&dataElementCount, outSum.get(), 1);
    }
    return dataElementCount;
}