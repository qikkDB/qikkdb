//
// Created by Martin Staňo on 2019-01-15.
//

#pragma once

#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <regex>
#include <string>
#include <mutex>
#include <unordered_map>
#include <map>
#include <condition_variable>
#include "../messages/QueryResponseMessage.pb.h"
#include "MemoryStream.h"
#include "../DataType.h"
#include "GroupByType.h"
#include "../QueryEngine/OrderByType.h"
#include "../IVariantArray.h"
#include "../QueryEngine/GPUCore/IGroupBy.h"
#include "../NativeGeoPoint.h"
#include "ParserExceptions.h"
#include "CpuSqlDispatcher.h"
#include "../ComplexPolygonFactory.h"
#include "../PointFactory.h"
#include "../QueryEngine/GPUCore/IOrderBy.h"

#ifndef NDEBUG
void AssertDeviceMatchesCurrentThread(int dispatcherThreadId_);
#endif

class Database;
struct InsertIntoStruct;

struct OrderByBlocks
{
    std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> ReconstructedOrderByOrderColumnBlocks;
    std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> ReconstructedOrderByOrderColumnNullBlocks;

    std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> ReconstructedOrderByRetColumnBlocks;
    std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> ReconstructedOrderByRetColumnNullBlocks;
};

class GPUOrderBy;

struct StringDataTypeComp
{
    explicit StringDataTypeComp(const std::string& s) : str(s)
    {
    }
    inline bool operator()(const std::pair<std::string, DataType>& p) const
    {
        return p.first == str;
    }

private:
    const std::string& str;
};

class GpuSqlDispatcher
{
private:
    struct PointerAllocation
    {
        std::uintptr_t GpuPtr;
        int32_t ElementCount;
        bool ShouldBeFreed;
        std::uintptr_t GpuNullMaskPtr;
    };
    static const std::string KEYS_SUFFIX;
    static const std::string NULL_SUFFIX;

    typedef int32_t (GpuSqlDispatcher::*DispatchFunction)();
    std::vector<DispatchFunction> dispatcherFunctions_;
    MemoryStream arguments_;
    int32_t blockIndex_;
    int64_t usedRegisterMemory_;
    const int64_t maxRegisterMemory_;
    int32_t dispatcherThreadId_;
    int32_t instructionPointer_;
    int32_t constPointCounter_;
    int32_t constPolygonCounter_;
    int32_t jmpInstructionPosition_;
    int32_t constStringCounter_;
    const std::shared_ptr<Database>& database_;
    std::string loadedTableName_;
    std::unordered_map<std::string, PointerAllocation> allocatedPointers_;
    std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIndices_;

    ColmnarDB::NetworkClient::Message::QueryResponseMessage responseMessage_;
    std::uintptr_t filter_;
    bool insideAggregation_;
    bool insideGroupBy_;
    bool usingGroupBy_;
    bool usingOrderBy_;
    bool usingJoin_;
    bool isLastBlockOfDevice_;
    bool isOverallLastBlock_;
    bool noLoad_;
    bool aborted_;
    int64_t loadNecessary_;
    std::vector<std::pair<std::string, DataType>> groupByColumns_;
    std::unordered_set<std::string> aggregatedRegisters_;
    std::unordered_set<std::string> registerLockList_;
    bool IsRegisterAllocated(const std::string& reg);
    std::pair<std::string, std::string> SplitColumnName(const std::string& colName);
    bool isValidCast(DataType fromType, DataType toType);
    std::vector<std::unique_ptr<IGroupBy>>& groupByTables_;
    CpuSqlDispatcher cpuDispatcher_;

    std::unique_ptr<InsertIntoStruct> insertIntoData_;
    std::unordered_map<std::string, std::vector<int8_t>> insertIntoNullMasks_;
    std::unique_ptr<IOrderBy> orderByTable_;
    std::vector<OrderByBlocks>& orderByBlocks_;

    std::unordered_map<std::string, std::unique_ptr<IVariantArray>> reconstructedOrderByColumnsMerged_;
    std::unordered_map<std::string, std::unique_ptr<int8_t[]>> reconstructedOrderByColumnsNullMerged_;

    std::unordered_map<int32_t, std::pair<std::string, OrderBy::Order>> orderByColumns_;
    std::vector<std::vector<int32_t>> orderByIndices_;


    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseOrFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseAndFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseXorFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseLeftShiftFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseRightShiftFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logarithmFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> arctangent2Functions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> concatFunctions;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> powerFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rootFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> pointFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> leftFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rightFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToIntFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToLongFunctions_;
    // static std::array<GpuSqlDispatcher::DispatchFunction,
    //		DataType::DATA_TYPE_SIZE> castToDateFunctions;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToFloatFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToDoubleFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToStringFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToPointFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToPolygonFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> castToInt8TFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> logicalNotFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> yearFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> monthFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> dayFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> hourFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> minuteFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> secondFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> minusFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> absoluteFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> sineFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> cosineFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> tangentFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> cotangentFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> arcsineFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> arccosineFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> arctangentFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> logarithm10Functions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> logarithmNaturalFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> exponentialFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> squareRootFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> squareFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> signFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> roundFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> ceilFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> floorFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> ltrimFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> rtrimFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> lowerFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> upperFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> reverseFunctions_;
    static std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> lenFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minAggregationFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxAggregationFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumAggregationFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countAggregationFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgAggregationFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minGroupByFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxGroupByFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumGroupByFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countGroupByFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgGroupByFunctions_;

    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> minGroupByMultiKeyFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> maxGroupByMultiKeyFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> sumGroupByMultiKeyFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> countGroupByMultiKeyFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> avgGroupByMultiKeyFunctions_;

    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> orderByFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> orderByReconstructFunctions_;

    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> retFunctions_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> groupByFunctions_;

    static DispatchFunction isNullFunction_;
    static DispatchFunction isNotNullFunction_;

    static DispatchFunction aggregationBeginFunction_;
    static DispatchFunction aggregationDoneFunction_;
    static DispatchFunction groupByBeginFunction_;
    static DispatchFunction groupByDoneFunction_;
    static DispatchFunction freeOrderByTableFunction_;
    static DispatchFunction orderByReconstructRetAllBlocksFunction_;
    static DispatchFunction filFunction_;
    static DispatchFunction whereEvaluationFunction_;
    static DispatchFunction lockRegisterFunction_;
    static DispatchFunction jmpFunction_;
    static DispatchFunction doneFunction_;
    static DispatchFunction showDatabasesFunction_;
    static DispatchFunction showTablesFunction_;
    static DispatchFunction showColumnsFunction_;
    static DispatchFunction createDatabaseFunction_;
    static DispatchFunction dropDatabaseFunction_;
    static DispatchFunction createTableFunction_;
    static DispatchFunction dropTableFunction_;
    static DispatchFunction alterTableFunction_;
    static DispatchFunction createIndexFunction_;
    static std::array<DispatchFunction, DataType::DATA_TYPE_SIZE> insertIntoFunctions_;
    static DispatchFunction insertIntoDoneFunction_;

    static int32_t groupByDoneCounter_;
    static int32_t orderByDoneCounter_;
    static int32_t deviceCountLimit_;
    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                             std::unique_ptr<int8_t[]>& data,
                                             int32_t dataSize);
    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                           std::unique_ptr<int32_t[]>& data,
                           int32_t dataSize);

    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                           std::unique_ptr<int64_t[]>& data,
                           int32_t dataSize);

    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                           std::unique_ptr<float[]>& data,
                           int32_t dataSize);

    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                           std::unique_ptr<double[]>& data,
                           int32_t dataSize);

    void InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                           std::unique_ptr<std::string[]>& data,
                           int32_t dataSize);

public:
    static std::mutex groupByMutex_;
    static std::mutex orderByMutex_;

    static std::condition_variable groupByCV_;
    static std::condition_variable orderByCV_;

    static void IncGroupByDoneCounter()
    {
        groupByDoneCounter_++;
    }

    static void IncOrderByDoneCounter()
    {
        orderByDoneCounter_++;
    }

    static bool IsGroupByDone()
    {
        return (groupByDoneCounter_ == deviceCountLimit_);
    }

    static bool IsOrderByDone()
    {
        return (orderByDoneCounter_ == deviceCountLimit_);
    }

    static void ResetGroupByCounters()
    {
        groupByDoneCounter_ = 0;
        deviceCountLimit_ = 0;
    }

    static void ResetOrderByCounters()
    {
        orderByDoneCounter_ = 0;
        deviceCountLimit_ = 0;
    }

    template <typename T>
    static std::pair<bool, T> AggregateOnCPU(std::string& operation, T number1, T number2)
    {
        if (operation == "MIN")
        {
            return std::make_pair(true, number1 < number2 ? number1 : number2);
        }
        else if (operation == "MAX")
        {
            return std::make_pair(true, number1 > number2 ? number1 : number2);
        }
        else if (operation == "SUM" || operation == "AVG" || operation == "COUNT")
        {
            return std::make_pair(true, number1 + number2);
        }
        else // Other operation (e.g. datetime)
        {
            return std::make_pair(false, T{0});
        }
    }

    static void MergePayload(const std::string& key,
                             ColmnarDB::NetworkClient::Message::QueryResponseMessage* responseMessage,
                             ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload);
    static void MergePayloadBitmask(const std::string& key,
                                    ColmnarDB::NetworkClient::Message::QueryResponseMessage* responseMessage,
                                    const std::string& nullMask);


    GpuSqlDispatcher(const std::shared_ptr<Database>& database,
                     std::vector<std::unique_ptr<IGroupBy>>& groupByTables,
                     std::vector<OrderByBlocks>& orderByBlocks,
                     int dispatcherThreadId);
    ~GpuSqlDispatcher();

    GpuSqlDispatcher(const GpuSqlDispatcher& dispatcher2) = delete;

    GpuSqlDispatcher& operator=(const GpuSqlDispatcher&) = delete;

    void CopyExecutionDataTo(GpuSqlDispatcher& other, CpuSqlDispatcher& sourceCpuDispatcher);

    void SetJoinIndices(std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIdx);

    void Execute(std::unique_ptr<google::protobuf::Message>& result, std::exception_ptr& exception);

    void Abort();

    const ColmnarDB::NetworkClient::Message::QueryResponseMessage& GetQueryResponseMessage();

    void AddGreaterFunction(DataType left, DataType right);

    void AddLessFunction(DataType left, DataType right);

    void AddGreaterEqualFunction(DataType left, DataType right);

    void AddLessEqualFunction(DataType left, DataType right);

    void AddEqualFunction(DataType left, DataType right);

    void AddNotEqualFunction(DataType left, DataType right);

    void AddLogicalAndFunction(DataType left, DataType right);

    void AddLogicalOrFunction(DataType left, DataType right);

    void AddMulFunction(DataType left, DataType right);

    void AddDivFunction(DataType left, DataType right);

    void AddAddFunction(DataType left, DataType right);

    void AddSubFunction(DataType left, DataType right);

    void AddModFunction(DataType left, DataType right);

    void AddBitwiseOrFunction(DataType left, DataType right);

    void AddBitwiseAndFunction(DataType left, DataType right);

    void AddBitwiseXorFunction(DataType left, DataType right);

    void AddBitwiseLeftShiftFunction(DataType left, DataType right);

    void AddBitwiseRightShiftFunction(DataType left, DataType right);

    void AddPointFunction(DataType left, DataType right);

    void AddContainsFunction(DataType left, DataType right);

    void AddIntersectFunction(DataType left, DataType right);

    void AddUnionFunction(DataType left, DataType right);

    void AddCastToIntFunction(DataType operand);

    void AddCastToLongFunction(DataType operand);

    void AddCastToDateFunction(DataType operand);

    void AddCastToFloatFunction(DataType operand);

    void AddCastToDoubleFunction(DataType operand);

    void AddCastToStringFunction(DataType operand);

    void AddCastToPointFunction(DataType operand);

    void AddCastToPolygonFunction(DataType operand);

    void AddCastToInt8TFunction(DataType operand);

    void AddLogicalNotFunction(DataType type);

    void AddIsNullFunction();

    void AddIsNotNullFunction();

    void AddMinusFunction(DataType type);

    void AddYearFunction(DataType type);

    void AddMonthFunction(DataType type);

    void AddDayFunction(DataType type);

    void AddHourFunction(DataType type);

    void AddMinuteFunction(DataType type);

    void AddSecondFunction(DataType type);

    void AddAbsoluteFunction(DataType type);

    void AddSineFunction(DataType type);

    void AddCosineFunction(DataType type);

    void AddTangentFunction(DataType type);

    void AddCotangentFunction(DataType type);

    void AddArcsineFunction(DataType type);

    void AddArccosineFunction(DataType type);

    void AddArctangentFunction(DataType type);

    void AddLogarithm10Function(DataType type);

    void AddLogarithmFunction(DataType number, DataType base);

    void AddArctangent2Function(DataType y, DataType x);

    void AddConcatFunction(DataType left, DataType right);

    void AddLeftFunction(DataType left, DataType right);

    void AddRightFunction(DataType left, DataType right);

    void AddLogarithmNaturalFunction(DataType type);

    void AddExponentialFunction(DataType type);

    void AddPowerFunction(DataType base, DataType exponent);

    void AddSquareRootFunction(DataType type);

    void AddSquareFunction(DataType type);

    void AddSignFunction(DataType type);

    void AddRoundFunction(DataType type);

    void AddFloorFunction(DataType type);

    void AddCeilFunction(DataType type);

    void AddLtrimFunction(DataType type);

    void AddRtrimFunction(DataType type);

    void AddLowerFunction(DataType type);

    void AddUpperFunction(DataType type);

    void AddReverseFunction(DataType type);

    void AddLenFunction(DataType type);

    void AddRootFunction(DataType base, DataType exponent);

    void AddMinFunction(DataType key, DataType value, GroupByType groupByType);

    void AddMaxFunction(DataType key, DataType value, GroupByType groupByType);

    void AddSumFunction(DataType key, DataType value, GroupByType groupByType);

    void AddCountFunction(DataType key, DataType value, GroupByType groupByType);

    void AddAvgFunction(DataType key, DataType value, GroupByType groupByType);

    void AddRetFunction(DataType type);

    void AddOrderByFunction(DataType type);

    void AddOrderByReconstructFunction(DataType type);

    void AddFreeOrderByTableFunction();

    void AddOrderByReconstructRetAllBlocksFunction();

    void AddLockRegisterFunction();

    void AddFilFunction();

    void AddWhereEvaluationFunction();

    void AddJmpInstruction();

    void AddDoneFunction();

    void AddShowDatabasesFunction();

    void AddShowTablesFunction();

    void AddShowColumnsFunction();

    void AddCreateDatabaseFunction();

    void AddDropDatabaseFunction();

    void AddCreateTableFunction();

    void AddDropTableFunction();

    void AddAlterTableFunction();

    void AddCreateIndexFunction();

    void AddInsertIntoFunction(DataType type);

    void AddInsertIntoDoneFunction();

    void AddGroupByFunction(DataType type);

    void AddGroupByBeginFunction();

    void AddGroupByDoneFunction();

    void AddAggregationBeginFunction();

    void AddAggregationDoneFunction();

    void AddBetweenFunction(DataType op1, DataType op2, DataType op3);

    void SetLoadedTableName(const std::string& tableName);

    static std::unordered_map<std::string, int32_t> linkTable;

    template <typename T>
    T* AllocateRegister(const std::string& reg, int32_t size, int8_t** nullPointerMask = nullptr)
    {
        T* gpuRegister;
        GPUMemory::alloc<T>(&gpuRegister, size);
        if (nullPointerMask)
        {
            int32_t bitMaskSize = ((size + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::alloc<int8_t>(nullPointerMask, bitMaskSize);
            InsertRegister(reg + NULL_SUFFIX, PointerAllocation{reinterpret_cast<std::uintptr_t>(*nullPointerMask),
                                                                bitMaskSize, true, 0});
            InsertRegister(reg, PointerAllocation{reinterpret_cast<std::uintptr_t>(gpuRegister), size, true,
                                                  reinterpret_cast<std::uintptr_t>(*nullPointerMask)});
        }
        else
        {
            InsertRegister(reg, PointerAllocation{reinterpret_cast<std::uintptr_t>(gpuRegister), size, true, 0});
        }

        usedRegisterMemory_ += size * sizeof(T);
        return gpuRegister;
    }

    void FillPolygonRegister(GPUMemory::GPUPolygon& polygonColumn,
                             const std::string& reg,
                             int32_t size,
                             bool useCache = false,
                             int8_t* nullMaskPtr = nullptr);

    /// Check if registerName is contained in allocatedPointers and if so, throw; if not, insert register
    void InsertRegister(const std::string& registerName, PointerAllocation registerValues);

    void FillStringRegister(GPUMemory::GPUString& stringColumn,
                            const std::string& reg,
                            int32_t size,
                            bool useCache = false,
                            int8_t* nullMaskPtr = nullptr);

    template <typename T>
    void AddCachedRegister(const std::string& reg, T* ptr, int32_t size, int8_t* nullMaskPtr = nullptr)
    {
        InsertRegister(reg, PointerAllocation{reinterpret_cast<std::uintptr_t>(ptr), size, false,
                                              reinterpret_cast<std::uintptr_t>(nullMaskPtr)});
    }

    template <typename T>
    int32_t LoadCol(std::string& colName);

    int32_t LoadColNullMask(std::string& colName);

    int32_t LoadTableBlockInfo(const std::string& tableName);

    size_t GetBlockSize();

    template <typename T>
    void FreeColumnIfRegister(const std::string& col)
    {
        if (usedRegisterMemory_ > maxRegisterMemory_ && !col.empty() && col.front() == '$' &&
            registerLockList_.find(col) == registerLockList_.end() &&
            allocatedPointers_.find(col) != allocatedPointers_.end())
        {
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Free: " << col << '\n';

            GPUMemory::free(reinterpret_cast<void*>(allocatedPointers_.at(col).GpuPtr));
            usedRegisterMemory_ -= allocatedPointers_.at(col).ElementCount * sizeof(T);
            allocatedPointers_.erase(col);

            if (allocatedPointers_.find(col + NULL_SUFFIX) != allocatedPointers_.end())
            {
                GPUMemory::free(reinterpret_cast<void*>(allocatedPointers_.at(col + NULL_SUFFIX).GpuPtr));
                usedRegisterMemory_ -= allocatedPointers_.at(col + NULL_SUFFIX).ElementCount * sizeof(int8_t);
                allocatedPointers_.erase(col + NULL_SUFFIX);
            }
        }
    }

    // TODO FreeColumnIfRegister<std::string> laso point and polygon
    void MergePayloadToSelfResponse(const std::string& key,
                                    ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                    const std::string& nullBitMaskString = "");

    GPUMemory::GPUPolygon InsertComplexPolygon(const std::string& databaseName,
                                               const std::string& colName,
                                               const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons,
                                               int32_t size,
                                               bool useCache = false,
                                               int8_t* nullMaskPtr = nullptr);
    GPUMemory::GPUString InsertString(const std::string& databaseName,
                                      const std::string& colName,
                                      const std::vector<std::string>& strings,
                                      int32_t size,
                                      bool useCache = false,
                                      int8_t* nullMaskPtr = nullptr);
    std::tuple<GPUMemory::GPUPolygon, int32_t, int8_t*> FindComplexPolygon(std::string colName);
    std::tuple<GPUMemory::GPUString, int32_t, int8_t*> FindStringColumn(const std::string& colName);
    void RewriteColumn(PointerAllocation& column, uintptr_t newPtr, int32_t newSize, int8_t* newNullMask);
    void RewriteStringColumn(const std::string& colName, GPUMemory::GPUString newStruct, int32_t newSize, int8_t* newNullMask);
    NativeGeoPoint* InsertConstPointGpu(ColmnarDB::Types::Point& point);
    GPUMemory::GPUPolygon InsertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon);
    GPUMemory::GPUString InsertConstStringGpu(const std::string& str, size_t size = 1);

    template <typename T>
    int32_t OrderByConst();

    template <typename T>
    int32_t OrderByCol();

    template <typename T>
    int32_t OrderByReconstructConst();

    template <typename T>
    int32_t OrderByReconstructCol();

    int32_t OrderByReconstructRetAllBlocks();

    template <typename T>
    int32_t RetConst();

    template <typename T>
    int32_t RetCol();

    int32_t AggregationBegin();

    int32_t AggregationDone();

    int32_t GroupByBegin();

    int32_t GroupByDone();

    int32_t FreeOrderByTable();

    int32_t LockRegister();

    int32_t Fil();

    int32_t WhereEvaluation();

    int32_t Jmp();

    int32_t Done();

    int32_t ShowDatabases();

    int32_t ShowTables();

    int32_t ShowColumns();

    int32_t CreateDatabase();

    int32_t DropDatabase();

    int32_t CreateTable();

    int32_t DropTable();

    int32_t AlterTable();

    int32_t CreateIndex();

    void CleanUpGpuPointers();


    //// FILTERS WITH FUNCTORS

    template <typename OP, typename T, typename U>
    int32_t FilterColConst();

    template <typename OP, typename T, typename U>
    int32_t FilterConstCol();

    template <typename OP, typename T, typename U>
    int32_t FilterColCol();

    template <typename OP, typename T, typename U>
    int32_t FilterConstConst();

    template <typename OP>
    int32_t FilterStringColConst();

    template <typename OP>
    int32_t FilterStringConstCol();

    template <typename OP>
    int32_t FilterStringColCol();

    template <typename OP>
    int32_t FilterStringConstConst();

    template <typename OP, typename T, typename U>
    int32_t LogicalColConst();

    template <typename OP, typename T, typename U>
    int32_t LogicalConstCol();

    template <typename OP, typename T, typename U>
    int32_t LogicalColCol();

    template <typename OP, typename T, typename U>
    int32_t LogicalConstConst();

    template <typename OP, typename T, typename U>
    int32_t ArithmeticColConst();

    template <typename OP, typename T, typename U>
    int32_t ArithmeticConstCol();

    template <typename OP, typename T, typename U>
    int32_t ArithmeticColCol();

    template <typename OP, typename T, typename U>
    int32_t ArithmeticConstConst();

    template <typename OP, typename T>
    int32_t ArithmeticUnaryCol();

    template <typename OP, typename T>
    int32_t ArithmeticUnaryConst();

    template <typename OP>
    int32_t StringUnaryCol();

    template <typename OP>
    int32_t StringUnaryConst();

    template <typename OP>
    int32_t StringUnaryNumericCol();

    template <typename OP>
    int32_t StringUnaryNumericConst();

    template <typename OP>
    int32_t StringBinaryColCol();

    template <typename OP>
    int32_t StringBinaryColConst();

    template <typename OP>
    int32_t StringBinaryConstCol();

    template <typename OP>
    int32_t StringBinaryConstConst();

    template <typename OP, typename T>
    int32_t StringBinaryNumericColCol();

    template <typename OP, typename T>
    int32_t StringBinaryNumericColConst();

    template <typename OP, typename T>
    int32_t StringBinaryNumericConstCol();

    template <typename OP, typename T>
    int32_t StringBinaryNumericConstConst();

    template <typename OP, typename R, typename T, typename U>
    int32_t AggregationGroupBy();

    template <typename OP, typename OUT, typename IN>
    int32_t AggregationCol();

    template <typename OP, typename T, typename U>
    int32_t AggregationConst();

    ////

    // point from columns

    template <typename T, typename U>
    int32_t PointColCol();

    template <typename T, typename U>
    int32_t PointColConst();

    template <typename T, typename U>
    int32_t PointConstCol();

    // contains

    template <typename T, typename U>
    int32_t ContainsColConst();

    template <typename T, typename U>
    int32_t ContainsConstCol();

    template <typename T, typename U>
    int32_t ContainsColCol();

    template <typename T, typename U>
    int32_t ContainsConstConst();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationColConst();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationConstCol();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationColCol();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationConstConst();

    template <typename OUT, typename IN>
    int32_t CastNumericCol();

    template <typename OUT, typename IN>
    int32_t CastNumericConst();

    template <typename OUT>
    int32_t CastStringCol();

    template <typename OUT>
    int32_t CastStringConst();

    int32_t CastPointCol();

    int32_t CastPointConst();

    int32_t CastPolygonCol();

    int32_t CastPolygonConst();

    int32_t Between();

    template <typename T>
    int32_t LogicalNotCol();

    template <typename T>
    int32_t LogicalNotConst();

    template <typename OP>
    int32_t NullMaskCol();

    template <typename OP>
    int32_t DateExtractCol();

    template <typename OP>
    int32_t DateExtractConst();

    template <typename T>
    int32_t GroupByCol();

    template <typename T>
    int32_t GroupByConst();

    template <typename T>
    int32_t InsertInto();

    int32_t InsertIntoDone();

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColConst()
    {
        U cnst = arguments_.Read<U>();
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstCol()
    {
        auto colName = arguments_.Read<std::string>();
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColCol()
    {
        auto colNameRight = arguments_.Read<std::string>();
        auto colNameLeft = arguments_.Read<std::string>();

        throw InvalidOperandsException(colNameLeft, colNameRight, std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstConst()
    {
        U cnstRight = arguments_.Read<U>();
        T cnstLeft = arguments_.Read<T>();

        throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string("operation"));
    }


    //// FUNCTOR ERROR HANDLERS

    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColConst()
    {
        U cnst = arguments_.Read<U>();
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
    }


    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstCol()
    {
        auto colName = arguments_.Read<std::string>();
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
    }


    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColCol()
    {
        auto colNameRight = arguments_.Read<std::string>();
        auto colNameLeft = arguments_.Read<std::string>();

        throw InvalidOperandsException(colNameLeft, colNameRight, std::string(typeid(OP).name()));
    }


    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstConst()
    {
        U cnstRight = arguments_.Read<U>();
        T cnstLeft = arguments_.Read<T>();

        throw InvalidOperandsException(std::string("cnst"), std::string("cnst"),
                                       std::string(typeid(OP).name()));
    }

    template <typename OP, typename T>
    int32_t InvalidOperandTypesErrorHandlerCol()
    {
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string(""), std::string(typeid(OP).name()));
    }

    template <typename OP, typename T>
    int32_t InvalidOperandTypesErrorHandlerConst()
    {
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string(typeid(OP).name()));
    }

    ////

    template <typename T>
    int32_t InvalidOperandTypesErrorHandlerCol()
    {
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string(""), std::string("operation"));
    }

    template <typename T>
    int32_t InvalidOperandTypesErrorHandlerConst()
    {
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string("operation"));
    }

    template <typename T>
    void AddArgument(T argument)
    {
        arguments_.Insert<T>(argument);
    }

private:
    template <typename OP, typename O, typename K, typename V>
    class GroupByHelper;

    template <typename OP, typename O, typename V>
    class GroupByHelper<OP, O, std::string, V>;
};

template <>
int32_t GpuSqlDispatcher::RetCol<ColmnarDB::Types::ComplexPolygon>();

template <>
int32_t GpuSqlDispatcher::RetCol<ColmnarDB::Types::Point>();

template <>
int32_t GpuSqlDispatcher::RetCol<std::string>();

template <>
int32_t GpuSqlDispatcher::RetConst<ColmnarDB::Types::ComplexPolygon>();

template <>
int32_t GpuSqlDispatcher::RetConst<ColmnarDB::Types::Point>();

template <>
int32_t GpuSqlDispatcher::RetConst<std::string>();

template <>
int32_t GpuSqlDispatcher::GroupByCol<std::string>();

template <>
int32_t GpuSqlDispatcher::InsertInto<ColmnarDB::Types::ComplexPolygon>();

template <>
int32_t GpuSqlDispatcher::InsertInto<ColmnarDB::Types::Point>();

template <>
int32_t GpuSqlDispatcher::LoadCol<ColmnarDB::Types::ComplexPolygon>(std::string& colName);

template <>
int32_t GpuSqlDispatcher::LoadCol<ColmnarDB::Types::Point>(std::string& colName);

template <>
int32_t GpuSqlDispatcher::LoadCol<std::string>(std::string& colName);

template <>
int32_t GpuSqlDispatcher::OrderByReconstructCol<std::string>();

template <>
int32_t GpuSqlDispatcher::OrderByReconstructCol<ColmnarDB::Types::Point>();

template <>
int32_t GpuSqlDispatcher::OrderByReconstructCol<ColmnarDB::Types::ComplexPolygon>();
