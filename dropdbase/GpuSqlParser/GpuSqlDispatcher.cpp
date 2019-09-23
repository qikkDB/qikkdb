//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlDispatcher.h"
#include "../QueryEngine/Context.h"
#include "../Types/ComplexPolygon.pb.h"
#include "../Types/Point.pb.h"
#include "ParserExceptions.h"
#include "../QueryEngine/Context.h"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../ComplexPolygonFactory.h"
#include "../StringFactory.h"
#include "../Database.h"
#include "../Table.h"
#include "LoadColHelper.h"
#include <any>
#include <string>
#include <unordered_map>
#include "InsertIntoStruct.h"

const std::string GpuSqlDispatcher::KEYS_SUFFIX = "_keys";
const std::string GpuSqlDispatcher::NULL_SUFFIX = "_nullMask";
const std::string GpuSqlDispatcher::RECONSTRUCTED_SUFFIX = "_reconstructed";

int32_t GpuSqlDispatcher::groupByDoneCounter_ = 0;
int32_t GpuSqlDispatcher::orderByDoneCounter_ = 0;

std::mutex GpuSqlDispatcher::groupByMutex_;
std::mutex GpuSqlDispatcher::orderByMutex_;

std::condition_variable GpuSqlDispatcher::groupByCV_;
std::condition_variable GpuSqlDispatcher::orderByCV_;

int32_t GpuSqlDispatcher::deviceCountLimit_;
std::unordered_map<std::string, int32_t> GpuSqlDispatcher::linkTable;

#ifndef NDEBUG
void AssertDeviceMatchesCurrentThread(int dispatcherThreadId_)
{
    int device = -1;
    cudaGetDevice(&device);
    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Current device for tid " << dispatcherThreadId_ << " is " << device << "\n";
    if (device != dispatcherThreadId_)
    {
        abort();
    }
}
#endif

GpuSqlDispatcher::GpuSqlDispatcher(const std::shared_ptr<Database>& database,
                                   std::vector<std::unique_ptr<IGroupBy>>& groupByTables,
                                   std::vector<OrderByBlocks>& orderByBlocks,
                                   int dispatcherThreadId)

: database_(database), blockIndex_(dispatcherThreadId), instructionPointer_(0),
  constPointCounter_(0), constPolygonCounter_(0), constStringCounter_(0), filter_(0),
  usedRegisterMemory_(0), maxRegisterMemory_(0), // TODO value from config e.g.
  groupByTables_(groupByTables), dispatcherThreadId_(dispatcherThreadId), insideAggregation_(false),
  insideGroupBy_(false), usingGroupBy_(false), usingOrderBy_(false), usingJoin_(false),
  isLastBlockOfDevice_(false), isOverallLastBlock_(false), noLoad_(true), aborted_(false),
  loadNecessary_(1), cpuDispatcher_(database), jmpInstructionPosition_(0),
  insertIntoData_(std::make_unique<InsertIntoStruct>()), joinIndices_(nullptr),
  orderByTable_(nullptr), orderByBlocks_(orderByBlocks), loadedTableName_("")
{
}

GpuSqlDispatcher::~GpuSqlDispatcher()
{
}

void GpuSqlDispatcher::SetLoadedTableName(const std::string& tableName)
{
    loadedTableName_ = tableName;
}

void GpuSqlDispatcher::CopyExecutionDataTo(GpuSqlDispatcher& other, CpuSqlDispatcher& sourceCpuDispatcher)
{
    other.dispatcherFunctions_ = dispatcherFunctions_;
    other.arguments_ = arguments_;
    other.jmpInstructionPosition_ = jmpInstructionPosition_;
    other.loadedTableName_ = loadedTableName_;
    sourceCpuDispatcher.CopyExecutionDataTo(other.cpuDispatcher_);
}

void GpuSqlDispatcher::SetJoinIndices(std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIdx)
{
    if (!joinIdx->empty())
    {
        joinIndices_ = joinIdx;
        usingJoin_ = true;
    }
}

/// Main execution loop of dispatcher
/// Iterates through all dispatcher functions in the operations array (filled from GpuSqlListener) and executes them
/// until running out of blocks
/// <param name="result">Response message to the SQL statement</param>
void GpuSqlDispatcher::Execute(std::unique_ptr<google::protobuf::Message>& result, std::exception_ptr& exception)
{
    try
    {
        Context& context = Context::getInstance();
        context.getCacheForCurrentDevice().setCurrentBlockIndex(blockIndex_);
        context.bindDeviceToContext(dispatcherThreadId_);
        context.getCacheForCurrentDevice().setCurrentBlockIndex(blockIndex_);

        LoadColHelper& loadColHelper = LoadColHelper::getInstance();

        int32_t err = 0;

        while (err == 0 && !aborted_)
        {

            err = (this->*dispatcherFunctions_[instructionPointer_++])();
#ifndef NDEBUG
            printf("tid:%d ip: %d \n", dispatcherThreadId_, instructionPointer_ - 1);
            AssertDeviceMatchesCurrentThread(dispatcherThreadId_);
#endif
            if (err)
            {
                if (err == 1)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info) << "Out of blocks." << '\n';
                }
                if (err == 2)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Show databases completed sucessfully" << '\n';
                }
                if (err == 3)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Show tables completed sucessfully" << '\n';
                }
                if (err == 4)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Show columns completed sucessfully" << '\n';
                }
                if (err == 5)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Insert into completed sucessfully" << '\n';
                }
                if (err == 6)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Create database_ completed sucessfully" << '\n';
                }
                if (err == 7)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Drop database_ completed sucessfully" << '\n';
                }
                if (err == 8)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Create table completed sucessfully" << '\n';
                }
                if (err == 9)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Drop table completed sucessfully" << '\n';
                }
                if (err == 10)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Alter table completed sucessfully" << '\n';
                }
                if (err == 11)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info)
                        << "Create index completed sucessfully" << '\n';
                }
                if (err == 12)
                {
                    CudaLogBoost::getInstance(CudaLogBoost::info) << "Load skipped" << '\n';
                    loadColHelper.countSkippedBlocks++;
                    err = 0;
                    continue;
                }
                break;
            }
        }
        result = std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>(
            std::move(responseMessage_));
    }
    catch (...)
    {
        exception = std::current_exception();
    }
    CleanUpGpuPointers();
}

void GpuSqlDispatcher::Abort()
{
    aborted_ = true;
}

const ColmnarDB::NetworkClient::Message::QueryResponseMessage& GpuSqlDispatcher::GetQueryResponseMessage()
{
    return responseMessage_;
}

void GpuSqlDispatcher::AddRetFunction(DataType type)
{
    dispatcherFunctions_.push_back(retFunctions_[type]);
}

void GpuSqlDispatcher::AddOrderByFunction(DataType type)
{
    dispatcherFunctions_.push_back(orderByFunctions_[type]);
}

void GpuSqlDispatcher::AddOrderByReconstructFunction(DataType type)
{
    dispatcherFunctions_.push_back(orderByReconstructFunctions_[type]);
}

void GpuSqlDispatcher::AddFreeOrderByTableFunction()
{
    dispatcherFunctions_.push_back(freeOrderByTableFunction_);
}

void GpuSqlDispatcher::AddOrderByReconstructRetAllBlocksFunction()
{
    dispatcherFunctions_.push_back(orderByReconstructRetAllBlocksFunction_);
}

void GpuSqlDispatcher::AddLockRegisterFunction()
{
    dispatcherFunctions_.push_back(lockRegisterFunction_);
}

void GpuSqlDispatcher::AddFilFunction()
{
    dispatcherFunctions_.push_back(filFunction_);
}

void GpuSqlDispatcher::AddWhereEvaluationFunction()
{
    dispatcherFunctions_.push_back(whereEvaluationFunction_);
}

void GpuSqlDispatcher::AddJmpInstruction()
{
    dispatcherFunctions_.push_back(jmpFunction_);
    jmpInstructionPosition_ = dispatcherFunctions_.size() - 1;
}

void GpuSqlDispatcher::AddDoneFunction()
{
    dispatcherFunctions_.push_back(doneFunction_);
}

void GpuSqlDispatcher::AddShowDatabasesFunction()
{
    dispatcherFunctions_.push_back(showDatabasesFunction_);
}

void GpuSqlDispatcher::AddShowTablesFunction()
{
    dispatcherFunctions_.push_back(showTablesFunction_);
}

void GpuSqlDispatcher::AddShowColumnsFunction()
{
    dispatcherFunctions_.push_back(showColumnsFunction_);
}

void GpuSqlDispatcher::AddCreateDatabaseFunction()
{
    dispatcherFunctions_.push_back(createDatabaseFunction_);
}

void GpuSqlDispatcher::AddDropDatabaseFunction()
{
    dispatcherFunctions_.push_back(dropDatabaseFunction_);
}

void GpuSqlDispatcher::AddCreateTableFunction()
{
    dispatcherFunctions_.push_back(createTableFunction_);
}

void GpuSqlDispatcher::AddDropTableFunction()
{
    dispatcherFunctions_.push_back(dropTableFunction_);
}

void GpuSqlDispatcher::AddAlterTableFunction()
{
    dispatcherFunctions_.push_back(alterTableFunction_);
}

void GpuSqlDispatcher::AddCreateIndexFunction()
{
    dispatcherFunctions_.push_back(createIndexFunction_);
}

void GpuSqlDispatcher::AddInsertIntoFunction(DataType type)
{
    dispatcherFunctions_.push_back(insertIntoFunctions_[type]);
}

void GpuSqlDispatcher::AddInsertIntoDoneFunction()
{
    dispatcherFunctions_.push_back(insertIntoDoneFunction_);
}

void GpuSqlDispatcher::AddGreaterFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(greaterFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddLessFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(lessFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddGreaterEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(greaterEqualFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddLessEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(lessEqualFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(equalFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddNotEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(notEqualFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddLogicalAndFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(logicalAndFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddLogicalOrFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(logicalOrFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddMulFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(mulFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddDivFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(divFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddAddFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(addFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::AddSubFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(subFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddModFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(modFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddBitwiseOrFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseOrFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddBitwiseAndFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseAndFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddBitwiseXorFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseXorFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddBitwiseLeftShiftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseLeftShiftFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddBitwiseRightShiftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseRightShiftFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddPointFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(pointFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddLogarithmFunction(DataType number, DataType base)
{
    dispatcherFunctions_.push_back(logarithmFunctions_[DataType::DATA_TYPE_SIZE * number + base]);
}

void GpuSqlDispatcher::AddArctangent2Function(DataType y, DataType x)
{
    dispatcherFunctions_.push_back(arctangent2Functions_[DataType::DATA_TYPE_SIZE * y + x]);
}

void GpuSqlDispatcher::AddRoundDecimalFunction(DataType y, DataType x)
{
    dispatcherFunctions_.push_back(roundDecimalFunctions_[DataType::DATA_TYPE_SIZE * y + x]);
}

void GpuSqlDispatcher::AddConcatFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(concatFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddLeftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(leftFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddRightFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(rightFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddPowerFunction(DataType base, DataType exponent)
{
    dispatcherFunctions_.push_back(powerFunctions_[DataType::DATA_TYPE_SIZE * base + exponent]);
}

void GpuSqlDispatcher::AddRootFunction(DataType base, DataType exponent)
{
    dispatcherFunctions_.push_back(rootFunctions_[DataType::DATA_TYPE_SIZE * base + exponent]);
}

void GpuSqlDispatcher::AddContainsFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(containsFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddIntersectFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(intersectFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddUnionFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(unionFunctions_[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::AddCastToIntFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToIntFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToLongFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToLongFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToDateFunction(DataType operand)
{
    // dispatcherFunctions_.push_back(castToDateFunctions[operand]);
}

void GpuSqlDispatcher::AddCastToFloatFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToFloatFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToDoubleFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToDoubleFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToStringFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToStringFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToPointFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToPointFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToPolygonFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToPolygonFunctions_[operand]);
}

void GpuSqlDispatcher::AddCastToInt8TFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToInt8TFunctions_[operand]);
}

void GpuSqlDispatcher::AddLogicalNotFunction(DataType type)
{
    dispatcherFunctions_.push_back(logicalNotFunctions_[type]);
}

void GpuSqlDispatcher::AddIsNullFunction()
{
    dispatcherFunctions_.push_back(isNullFunction_);
}

void GpuSqlDispatcher::AddIsNotNullFunction()
{
    dispatcherFunctions_.push_back(isNotNullFunction_);
}

void GpuSqlDispatcher::AddMinusFunction(DataType type)
{
    dispatcherFunctions_.push_back(minusFunctions_[type]);
}

void GpuSqlDispatcher::AddYearFunction(DataType type)
{
    dispatcherFunctions_.push_back(yearFunctions_[type]);
}

void GpuSqlDispatcher::AddMonthFunction(DataType type)
{
    dispatcherFunctions_.push_back(monthFunctions_[type]);
}

void GpuSqlDispatcher::AddDayFunction(DataType type)
{
    dispatcherFunctions_.push_back(dayFunctions_[type]);
}

void GpuSqlDispatcher::AddHourFunction(DataType type)
{
    dispatcherFunctions_.push_back(hourFunctions_[type]);
}

void GpuSqlDispatcher::AddMinuteFunction(DataType type)
{
    dispatcherFunctions_.push_back(minuteFunctions_[type]);
}

void GpuSqlDispatcher::AddSecondFunction(DataType type)
{
    dispatcherFunctions_.push_back(secondFunctions_[type]);
}

void GpuSqlDispatcher::AddAbsoluteFunction(DataType type)
{
    dispatcherFunctions_.push_back(absoluteFunctions_[type]);
}

void GpuSqlDispatcher::AddSineFunction(DataType type)
{
    dispatcherFunctions_.push_back(sineFunctions_[type]);
}

void GpuSqlDispatcher::AddCosineFunction(DataType type)
{
    dispatcherFunctions_.push_back(cosineFunctions_[type]);
}

void GpuSqlDispatcher::AddTangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(tangentFunctions_[type]);
}

void GpuSqlDispatcher::AddCotangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(cotangentFunctions_[type]);
}

void GpuSqlDispatcher::AddArcsineFunction(DataType type)
{
    dispatcherFunctions_.push_back(arcsineFunctions_[type]);
}

void GpuSqlDispatcher::AddArccosineFunction(DataType type)
{
    dispatcherFunctions_.push_back(arccosineFunctions_[type]);
}

void GpuSqlDispatcher::AddArctangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(arctangentFunctions_[type]);
}

void GpuSqlDispatcher::AddLogarithm10Function(DataType type)
{
    dispatcherFunctions_.push_back(logarithm10Functions_[type]);
}

void GpuSqlDispatcher::AddLogarithmNaturalFunction(DataType type)
{
    dispatcherFunctions_.push_back(logarithmNaturalFunctions_[type]);
}

void GpuSqlDispatcher::AddExponentialFunction(DataType type)
{
    dispatcherFunctions_.push_back(exponentialFunctions_[type]);
}

void GpuSqlDispatcher::AddSquareFunction(DataType type)
{
    dispatcherFunctions_.push_back(squareFunctions_[type]);
}

void GpuSqlDispatcher::AddSquareRootFunction(DataType type)
{
    dispatcherFunctions_.push_back(squareRootFunctions_[type]);
}

void GpuSqlDispatcher::AddSignFunction(DataType type)
{
    dispatcherFunctions_.push_back(signFunctions_[type]);
}

void GpuSqlDispatcher::AddRoundFunction(DataType type)
{
    dispatcherFunctions_.push_back(roundFunctions_[type]);
}

void GpuSqlDispatcher::AddFloorFunction(DataType type)
{
    dispatcherFunctions_.push_back(floorFunctions_[type]);
}

void GpuSqlDispatcher::AddCeilFunction(DataType type)
{
    dispatcherFunctions_.push_back(ceilFunctions_[type]);
}

void GpuSqlDispatcher::AddLtrimFunction(DataType type)
{
    dispatcherFunctions_.push_back(ltrimFunctions_[type]);
}

void GpuSqlDispatcher::AddRtrimFunction(DataType type)
{
    dispatcherFunctions_.push_back(rtrimFunctions_[type]);
}

void GpuSqlDispatcher::AddLowerFunction(DataType type)
{
    dispatcherFunctions_.push_back(lowerFunctions_[type]);
}

void GpuSqlDispatcher::AddUpperFunction(DataType type)
{
    dispatcherFunctions_.push_back(upperFunctions_[type]);
}

void GpuSqlDispatcher::AddReverseFunction(DataType type)
{
    dispatcherFunctions_.push_back(reverseFunctions_[type]);
}

void GpuSqlDispatcher::AddLenFunction(DataType type)
{
    dispatcherFunctions_.push_back(lenFunctions_[type]);
}

void GpuSqlDispatcher::AddMinFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = minAggregationFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = minGroupByFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = minGroupByMultiKeyFunctions_[value];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddMaxFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = maxAggregationFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = maxGroupByFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = maxGroupByMultiKeyFunctions_[value];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddSumFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = sumAggregationFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = sumGroupByFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = sumGroupByMultiKeyFunctions_[value];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddCountFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = countAggregationFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = countGroupByFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = countGroupByMultiKeyFunctions_[value];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddAvgFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = avgAggregationFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = avgGroupByFunctions_[DataType::DATA_TYPE_SIZE * key + value];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = avgGroupByMultiKeyFunctions_[value];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddGroupByFunction(DataType type)
{
    dispatcherFunctions_.push_back(groupByFunctions_[type]);
}

void GpuSqlDispatcher::AddGroupByBeginFunction()
{
    dispatcherFunctions_.push_back(groupByBeginFunction_);
}

void GpuSqlDispatcher::AddGroupByDoneFunction()
{
    dispatcherFunctions_.push_back(groupByDoneFunction_);
}

void GpuSqlDispatcher::AddAggregationBeginFunction()
{
    dispatcherFunctions_.push_back(aggregationBeginFunction_);
}

void GpuSqlDispatcher::AddAggregationDoneFunction()
{
    dispatcherFunctions_.push_back(aggregationDoneFunction_);
}

void GpuSqlDispatcher::AddBetweenFunction(DataType op1, DataType op2, DataType op3)
{
    // TODO: Between
}

void GpuSqlDispatcher::FillPolygonRegister(GPUMemory::GPUPolygon& polygonColumn,
                                           const std::string& reg,
                                           int32_t size,
                                           bool useCache,
                                           int8_t* nullMaskPtr)
{
    InsertRegister(reg + "_polyPoints",
                   PointerAllocation{reinterpret_cast<uintptr_t>(polygonColumn.polyPoints), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_pointIdx",
                   PointerAllocation{reinterpret_cast<uintptr_t>(polygonColumn.pointIdx), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_polyIdx",
                   PointerAllocation{reinterpret_cast<uintptr_t>(polygonColumn.polyIdx), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
}

void GpuSqlDispatcher::InsertRegister(const std::string& registerName, PointerAllocation registerValues)
{
    if (allocatedPointers_.find(registerName) == allocatedPointers_.end())
    {
        allocatedPointers_.insert({registerName, registerValues});
    }
    else
    {
        throw std::runtime_error("Attempt to overwrite existing register \"" + registerName + "\"");
    }
}

void GpuSqlDispatcher::FillStringRegister(GPUMemory::GPUString& stringColumn,
                                          const std::string& reg,
                                          int32_t size,
                                          bool useCache,
                                          int8_t* nullMaskPtr)
{
    InsertRegister(reg + "_stringIndices",
                   PointerAllocation{reinterpret_cast<uintptr_t>(stringColumn.stringIndices), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
    InsertRegister(reg + "_allChars",
                   PointerAllocation{reinterpret_cast<uintptr_t>(stringColumn.allChars), size,
                                     !useCache, reinterpret_cast<uintptr_t>(nullMaskPtr)});
}

int32_t GpuSqlDispatcher::LoadColNullMask(std::string& colName)
{
    if (allocatedPointers_.find(colName + NULL_SUFFIX) == allocatedPointers_.end() &&
        !colName.empty() && colName.front() != '$')
    {
        CudaLogBoost::getInstance(CudaLogBoost::info) << "LoadNullMask: " << colName << '\n';

        // split colName to table and column name
        const size_t endOfPolyIdx = colName.find(".");
        const std::string table = colName.substr(0, endOfPolyIdx);
        const std::string column = colName.substr(endOfPolyIdx + 1);

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

        auto blockNullMask =
            database_->GetTables().at(table).GetColumns().at(column)->GetNullBitMaskForBlock(blockIndex_);
        size_t blockNullMaskSize =
            (std::get<1>(blockNullMask) + 8 * sizeof(int8_t) - 1) / (8 * sizeof(int8_t));

        auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
            database_->GetName(), colName + NULL_SUFFIX, blockIndex_, blockNullMaskSize);
        if (!std::get<2>(cacheEntry))
        {
            GPUMemory::copyHostToDevice(std::get<0>(cacheEntry), std::get<0>(blockNullMask), blockNullMaskSize);
        }
        AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheEntry), std::get<1>(blockNullMask));

        noLoad_ = false;
    }
    return 0;
}

GPUMemory::GPUPolygon
GpuSqlDispatcher::InsertComplexPolygon(const std::string& databaseName,
                                       const std::string& colName,
                                       const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons,
                                       int32_t size,
                                       bool useCache,
                                       int8_t* nullMaskPtr)
{
    if (useCache)
    {
        if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyPoints",
                                                                             blockIndex_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointIdx",
                                                                             blockIndex_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyIdx", blockIndex_))
        {
            GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
            GPUMemory::GPUPolygon polygon;
            polygon.polyPoints = std::get<0>(
                cache.getColumn<NativeGeoPoint>(databaseName, colName + "_polyPoints", blockIndex_, size));
            polygon.pointIdx =
                std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_pointIdx", blockIndex_, size));
            polygon.polyIdx =
                std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_polyIdx", blockIndex_, size));

            FillPolygonRegister(polygon, colName, size, useCache, nullMaskPtr);

            return polygon;
        }
        else
        {
            GPUMemory::GPUPolygon polygon =
                ComplexPolygonFactory::PrepareGPUPolygon(polygons, databaseName, colName, blockIndex_);
            FillPolygonRegister(polygon, colName, size, useCache, nullMaskPtr);
            return polygon;
        }
    }
    else
    {
        GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons);
        FillPolygonRegister(polygon, colName, size, useCache, nullMaskPtr);
        return polygon;
    }
}

GPUMemory::GPUString GpuSqlDispatcher::InsertString(const std::string& databaseName,
                                                    const std::string& colName,
                                                    const std::vector<std::string>& strings,
                                                    int32_t size,
                                                    bool useCache,
                                                    int8_t* nullMaskPtr)
{
    if (useCache)
    {
        if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_stringIndices",
                                                                             blockIndex_) &&
            Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_allChars",
                                                                             blockIndex_))
        {
            GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
            GPUMemory::GPUString gpuString;
            gpuString.stringIndices = std::get<0>(
                cache.getColumn<int64_t>(databaseName, colName + "_stringIndices", blockIndex_, size));
            gpuString.allChars =
                std::get<0>(cache.getColumn<char>(databaseName, colName + "_allChars", blockIndex_, size));
            FillStringRegister(gpuString, colName, size, useCache, nullMaskPtr);
            return gpuString;
        }
        else
        {
            GPUMemory::GPUString gpuString =
                StringFactory::PrepareGPUString(strings, databaseName, colName, blockIndex_);
            FillStringRegister(gpuString, colName, size, useCache, nullMaskPtr);
            return gpuString;
        }
    }
    else
    {
        GPUMemory::GPUString gpuString = StringFactory::PrepareGPUString(strings);
        FillStringRegister(gpuString, colName, size, useCache, nullMaskPtr);
        return gpuString;
    }
}

std::tuple<GPUMemory::GPUPolygon, int32_t, int8_t*> GpuSqlDispatcher::FindComplexPolygon(std::string colName)
{
    GPUMemory::GPUPolygon polygon;
    int32_t size = allocatedPointers_.at(colName + "_polyPoints").ElementCount;

    polygon.polyPoints =
        reinterpret_cast<NativeGeoPoint*>(allocatedPointers_.at(colName + "_polyPoints").GpuPtr);
    polygon.pointIdx = reinterpret_cast<int32_t*>(allocatedPointers_.at(colName + "_pointIdx").GpuPtr);
    polygon.polyIdx = reinterpret_cast<int32_t*>(allocatedPointers_.at(colName + "_polyIdx").GpuPtr);

    return std::make_tuple(polygon, size,
                           reinterpret_cast<int8_t*>(allocatedPointers_.at(colName + "_polyPoints").GpuNullMaskPtr));
}

std::tuple<GPUMemory::GPUString, int32_t, int8_t*> GpuSqlDispatcher::FindStringColumn(const std::string& colName)
{
    GPUMemory::GPUString gpuString;
    int32_t size = allocatedPointers_.at(colName + "_stringIndices").ElementCount;
    gpuString.stringIndices =
        reinterpret_cast<int64_t*>(allocatedPointers_.at(colName + "_stringIndices").GpuPtr);
    gpuString.allChars = reinterpret_cast<char*>(allocatedPointers_.at(colName + "_allChars").GpuPtr);
    return std::make_tuple(gpuString, size,
                           reinterpret_cast<int8_t*>(
                               allocatedPointers_.at(colName + "_stringIndices").GpuNullMaskPtr));
}

void GpuSqlDispatcher::RewriteColumn(PointerAllocation& column,
                                     uintptr_t reconstructedReg,
                                     int32_t reconstructedSize,
                                     int8_t* reconstructedNullMask)
{
    if (filter_) // If where mask was used, new buffers were allocated, need to free old GpuPtr
    {
        if (column.ShouldBeFreed) // should be freed if it is not cached - if it is temp register like "YEAR(col)"
        {
            GPUMemory::free(reinterpret_cast<void*>(column.GpuPtr));
            // Do not free null mask because it is stored also as col_nullmask in allocated pointers
        }
        else // If original column was cachced, after rewrite the new pointer will need to be freed
        {
            column.ShouldBeFreed = true; // enable future free in cleanupGpuPointers
        }
    }

    // Now rewrite the pointer in the register (correct because the pointer is either freed or stored in chache)
    column.GpuPtr = reconstructedReg;
    column.ElementCount = reconstructedSize;
    column.GpuNullMaskPtr = reinterpret_cast<uintptr_t>(reconstructedNullMask);
}

void GpuSqlDispatcher::RewriteStringColumn(const std::string& colName,
                                           GPUMemory::GPUString newStruct,
                                           int32_t newElementCount,
                                           int8_t* newNullMask)
{
    if (filter_)
    {
        const auto column = FindStringColumn(colName);
        GPUMemory::free(std::get<0>(column));
        // Do not free null mask (std::get<2>) because it is stored also as col_nullmask in allocated pointers
    }

    // Find corresponding pointers
    PointerAllocation& stringIndices = allocatedPointers_.at(colName + "_stringIndices");
    PointerAllocation& allChars = allocatedPointers_.at(colName + "_allChars");

    // Rewrite stringIndices
    stringIndices.GpuPtr = reinterpret_cast<uintptr_t>(newStruct.stringIndices);
    stringIndices.ElementCount = newElementCount;
    stringIndices.GpuNullMaskPtr = reinterpret_cast<uintptr_t>(newNullMask);

    // Rewrite allChars
    allChars.GpuPtr = reinterpret_cast<uintptr_t>(newStruct.allChars);
    allChars.ElementCount = newElementCount;
    allChars.GpuNullMaskPtr = reinterpret_cast<uintptr_t>(newNullMask);
}

GPUMemory::GPUString GpuSqlDispatcher::InsertConstStringGpu(const std::string& str, size_t size)
{
    std::vector<std::string> strings(size, str);
    std::string name = "constString" + std::to_string(constStringCounter_);
    constStringCounter_++;
    return InsertString(database_->GetName(), name, strings, 1);
}

/// Clears all allocated buffers
/// Resets memory stream reading index to prepare for execution on the next block of data
void GpuSqlDispatcher::CleanUpGpuPointers()
{
    usingGroupBy_ = false;
    arguments_.Reset();
    for (auto& ptr : allocatedPointers_)
    {
        if (ptr.second.GpuPtr != 0 && ptr.second.ShouldBeFreed)
        {
            GPUMemory::free(reinterpret_cast<void*>(ptr.second.GpuPtr));
        }
    }
    usedRegisterMemory_ = 0;
    filter_ = 0;
    usingGroupBy_ = false;
    aggregatedRegisters_.clear();
    allocatedPointers_.clear();
    orderByTable_.reset();
}


/// Implementation of FIL operation
/// Marks WHERE clause result register as the filtering register
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::Fil()
{
    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Filter: " << reg << '\n';
    filter_ = allocatedPointers_.at(reg).GpuPtr;
    return 0;
}

int32_t GpuSqlDispatcher::WhereEvaluation()
{
    loadNecessary_ = 1; // usingJoin_ ? 1 : cpuDispatcher_.Execute(blockIndex_);
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Where load evaluation: " << loadNecessary_ << '\n';
    return 0;
}


/// Implementation of JMP operation
/// Determines next block index to process by this instance of dispatcher based on CUDA device count
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::Jmp()
{
    Context& context = Context::getInstance();

    if (noLoad_ && loadNecessary_ != 0)
    {
        CleanUpGpuPointers();
        return 0;
    }

    if (!isLastBlockOfDevice_)
    {
        blockIndex_ += context.getDeviceCount();
        context.getCacheForCurrentDevice().setCurrentBlockIndex(blockIndex_);
        instructionPointer_ = 0;
        CleanUpGpuPointers();
        return 0;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Jump" << '\n';
    return 0;
}


/// Implementation of DONE operation
/// Clears all allocated temporary result buffers
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::Done()
{
    CleanUpGpuPointers();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Done" << '\n';
    return 1;
}

/// Implementation of SHOW DATABASES operation
/// Inserts database names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::ShowDatabases()
{
    auto databases_map = Database::GetDatabaseNames();
    std::unique_ptr<std::string[]> outData(new std::string[databases_map.size()]);

    int i = 0;
    for (auto& database : databases_map)
    {
        outData[i++] = database;
    }

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    InsertIntoPayload(payload, outData, databases_map.size());
    MergePayloadToSelfResponse("Databases", payload);

    return 2;
}


/// Implementation of SHOW TABLES operation
/// Inserts table names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::ShowTables()
{
    std::string db = arguments_.Read<std::string>();
    std::shared_ptr<Database> database = Database::GetDatabaseByName(db);

    std::unique_ptr<std::string[]> outData(new std::string[database->GetTables().size()]);
    auto& tables_map = database->GetTables();

    int i = 0;
    for (auto& tableName : tables_map)
    {
        outData[i++] = tableName.first;
    }

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
    InsertIntoPayload(payload, outData, tables_map.size());
    MergePayloadToSelfResponse(db, payload);

    return 3;
}

/// Implementation of SHOW COLUMN operation
/// Inserts column names and their types to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::ShowColumns()
{
    std::string db = arguments_.Read<std::string>();
    std::string tab = arguments_.Read<std::string>();

    std::shared_ptr<Database> database = Database::GetDatabaseByName(db);
    auto& table = database_->GetTables().at(tab);

    auto& columns_map = table.GetColumns();
    // std::vector<std::string> columns;
    std::unique_ptr<std::string[]> outDataName(new std::string[table.GetColumns().size()]);
    std::unique_ptr<std::string[]> outDataType(new std::string[table.GetColumns().size()]);

    int i = 0;
    for (auto& column : columns_map)
    {
        outDataName[i] = column.first;
        outDataType[i] = ::GetStringFromColumnDataType(column.second.get()->GetColumnType());
        i++;
    }

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadName;
    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadType;
    InsertIntoPayload(payloadName, outDataName, columns_map.size());
    InsertIntoPayload(payloadType, outDataType, columns_map.size());
    MergePayloadToSelfResponse(tab + "_columns", payloadName);
    MergePayloadToSelfResponse(tab + "_types", payloadType);
    return 4;
}

int32_t GpuSqlDispatcher::CreateDatabase()
{
    std::string newDbName = arguments_.Read<std::string>();
    int32_t newDbBlockSize = arguments_.Read<int32_t>();
    std::shared_ptr<Database> newDb = std::make_shared<Database>(newDbName.c_str(), newDbBlockSize);
    Database::AddToInMemoryDatabaseList(newDb);
    return 6;
}

int32_t GpuSqlDispatcher::DropDatabase()
{
    std::string dbName = arguments_.Read<std::string>();
    Database::GetDatabaseByName(dbName)->DeleteDatabaseFromDisk();
    Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
    return 7;
}

int32_t GpuSqlDispatcher::CreateTable()
{
    std::unordered_map<std::string, DataType> newColumns;
    std::unordered_map<std::string, std::vector<std::string>> newIndices;

    std::string newTableName = arguments_.Read<std::string>();

    int32_t newColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < newColumnsCount; i++)
    {
        std::string newColumnName = arguments_.Read<std::string>();
        int32_t newColumnDataType = arguments_.Read<int32_t>();
        newColumns.insert({newColumnName, static_cast<DataType>(newColumnDataType)});
    }

    std::vector<std::string> allIndexColumns;

    int32_t newIndexCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < newIndexCount; i++)
    {
        std::string newIndexName = arguments_.Read<std::string>();
        int32_t newIndexColumnCount = arguments_.Read<int32_t>();
        std::vector<std::string> newIndexColumns;

        for (int32_t j = 0; j < newIndexColumnCount; j++)
        {
            std::string newIndexColumn = arguments_.Read<std::string>();
            newIndexColumns.push_back(newIndexColumn);
            if (std::find(allIndexColumns.begin(), allIndexColumns.end(), newIndexColumn) ==
                allIndexColumns.end())
            {
                allIndexColumns.push_back(newIndexColumn);
            }
        }
        newIndices.insert({newIndexName, newIndexColumns});
    }

    database_->CreateTable(newColumns, newTableName.c_str()).SetSortingColumns(allIndexColumns);
    return 8;
}

int32_t GpuSqlDispatcher::DropTable()
{
    std::string tableName = arguments_.Read<std::string>();
    database_->GetTables().erase(tableName);
    database_->DeleteTableFromDisk(tableName.c_str());
    return 9;
}

int32_t GpuSqlDispatcher::AlterTable()
{
    std::string tableName = arguments_.Read<std::string>();

    int32_t addColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < addColumnsCount; i++)
    {
        std::string addColumnName = arguments_.Read<std::string>();
        int32_t addColumnDataType = arguments_.Read<int32_t>();
        database_->GetTables().at(tableName).CreateColumn(addColumnName.c_str(),
                                                          static_cast<DataType>(addColumnDataType));
        database_->GetTables().at(tableName).InsertNullDataIntoNewColumn(addColumnName);
    }

    int32_t dropColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < dropColumnsCount; i++)
    {
        std::string dropColumnName = arguments_.Read<std::string>();
        database_->GetTables().at(tableName).EraseColumn(dropColumnName);
        database_->DeleteColumnFromDisk(tableName.c_str(), dropColumnName.c_str());
    }

    int32_t alterColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < alterColumnsCount; i++)
    {
        std::string alterColumnName = arguments_.Read<std::string>();
        int32_t alterColumnDataType = arguments_.Read<int32_t>();

        auto originType =
            database_->GetTables().at(tableName).GetColumns().at(alterColumnName)->GetColumnType();
        if (isValidCast(originType, static_cast<DataType>(alterColumnDataType)) &&
            originType != static_cast<DataType>(alterColumnDataType))
        {
            database_->GetTables().at(tableName).CreateColumn((alterColumnName + "_temp").c_str(),
                                                              static_cast<DataType>(alterColumnDataType));
            auto oldColumn = database_->GetTables().at(tableName).GetColumns().at(alterColumnName).get();
            auto newColumn =
                database_->GetTables().at(tableName).GetColumns().at(alterColumnName + "_temp").get();

            switch (originType)
            {
            case COLUMN_INT:
            {
                dynamic_cast<ColumnBase<int32_t>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_LONG:
            {
                dynamic_cast<ColumnBase<int64_t>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_FLOAT:
            {
                dynamic_cast<ColumnBase<float>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_DOUBLE:
            {
                dynamic_cast<ColumnBase<double>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_POINT:
            {
                dynamic_cast<ColumnBase<ColmnarDB::Types::Point>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_POLYGON:
            {
                dynamic_cast<ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_STRING:
            {
                dynamic_cast<ColumnBase<std::string>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }

            case COLUMN_INT8_T:
            {
                dynamic_cast<ColumnBase<int8_t>*>(oldColumn)->CopyDataToColumn(newColumn);
                break;
            }
            default:
                throw std::runtime_error("Attempt to execute unsupported column type conversion.");
                break;
            }

            database_->GetTables().at(tableName).EraseColumn(alterColumnName);
            database_->GetTables().at(tableName).RenameColumn(alterColumnName + "_temp", alterColumnName);
        }
    }
    return 10;
}

int32_t GpuSqlDispatcher::CreateIndex()
{
    std::string indexName = arguments_.Read<std::string>();
    std::string tableName = arguments_.Read<std::string>();
    std::vector<std::string> sortingColumns;

    for (auto& column : database_->GetTables().at(tableName).GetSortingColumns())
    {
        sortingColumns.push_back(column);
    }

    int32_t indexColumnCount = arguments_.Read<int32_t>();
    for (int i = 0; i < indexColumnCount; i++)
    {
        std::string indexColumn = arguments_.Read<std::string>();
        if (std::find(sortingColumns.begin(), sortingColumns.end(), indexColumn) == sortingColumns.end())
        {
            sortingColumns.push_back(indexColumn);
        }
    }

    database_->GetTables().at(tableName).SetSortingColumns(sortingColumns);

    return 11;
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<int8_t[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_intpayload()->add_intdata(data[i]);
    }
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<int32_t[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_intpayload()->add_intdata(data[i]);
    }
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<int64_t[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_int64payload()->add_int64data(data[i]);
    }
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<float[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_floatpayload()->add_floatdata(data[i]);
    }
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<double[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_doublepayload()->add_doubledata(data[i]);
    }
}

void GpuSqlDispatcher::InsertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                         std::unique_ptr<std::string[]>& data,
                                         int32_t dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        payload.mutable_stringpayload()->add_stringdata(data[i]);
    }
}

void GpuSqlDispatcher::MergePayloadBitmask(const std::string& key,
                                           ColmnarDB::NetworkClient::Message::QueryResponseMessage* responseMessage,
                                           const std::string& nullMask)
{
    if (responseMessage->nullbitmasks().find(key) == responseMessage->nullbitmasks().end())
    {
        responseMessage->mutable_nullbitmasks()->insert({key, nullMask});
    }
    else // If there is payload with existing key, merge or aggregate according to key
    {
        responseMessage->mutable_nullbitmasks()->at(key) += nullMask;
    }
}

void GpuSqlDispatcher::MergePayload(const std::string& trimmedKey,
                                    ColmnarDB::NetworkClient::Message::QueryResponseMessage* responseMessage,
                                    ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload)
{
    // If there is payload with new key
    if (responseMessage->payloads().find(trimmedKey) == responseMessage->payloads().end())
    {
        responseMessage->mutable_payloads()->insert({trimmedKey, payload});
    }
    else // If there is payload with existing key, merge or aggregate according to key
    {
        // Find index of parenthesis (for finding out if it is aggregation function)
        size_t keyParensIndex = trimmedKey.find('(');

        bool aggregationOperationFound = false;
        // If no function is used
        if (keyParensIndex == std::string::npos)
        {
            aggregationOperationFound = false;
        }
        else
        {
            // Get operation name
            std::string operation = trimmedKey.substr(0, keyParensIndex);
            // To upper case
            for (auto& c : operation)
            {
                c = toupper(c);
            }
            // Switch according to data type of payload (=column)
            switch (payload.payload_case())
            {
            case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
            {
                std::pair<bool, int32_t> result = AggregateOnCPU<int32_t>(
                    operation, payload.intpayload().intdata()[0],
                    responseMessage->mutable_payloads()->at(trimmedKey).intpayload().intdata()[0]);
                aggregationOperationFound = result.first;
                if (aggregationOperationFound)
                {
                    responseMessage->mutable_payloads()
                        ->at(trimmedKey)
                        .mutable_intpayload()
                        ->set_intdata(0, result.second);
                }
                break;
            }
            case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
            {
                std::pair<bool, int64_t> result = AggregateOnCPU<int64_t>(
                    operation, payload.int64payload().int64data()[0],
                    responseMessage->payloads().at(trimmedKey).int64payload().int64data()[0]);
                aggregationOperationFound = result.first;
                if (aggregationOperationFound)
                {
                    responseMessage->mutable_payloads()
                        ->at(trimmedKey)
                        .mutable_int64payload()
                        ->set_int64data(0, result.second);
                }
                break;
            }
            case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
            {
                std::pair<bool, float> result = AggregateOnCPU<float>(
                    operation, payload.floatpayload().floatdata()[0],
                    responseMessage->mutable_payloads()->at(trimmedKey).floatpayload().floatdata()[0]);
                aggregationOperationFound = result.first;
                if (aggregationOperationFound)
                {
                    responseMessage->mutable_payloads()
                        ->at(trimmedKey)
                        .mutable_floatpayload()
                        ->set_floatdata(0, result.second);
                }
                break;
            }
            case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
            {
                std::pair<bool, double> result = AggregateOnCPU<double>(
                    operation, payload.doublepayload().doubledata()[0],
                    responseMessage->mutable_payloads()->at(trimmedKey).doublepayload().doubledata()[0]);
                aggregationOperationFound = result.first;
                if (aggregationOperationFound)
                {
                    responseMessage->mutable_payloads()
                        ->at(trimmedKey)
                        .mutable_doublepayload()
                        ->set_doubledata(0, result.second);
                }
                break;
            }
            default:
                // This case is taken even without aggregation functions, because Points are
                // considered functions for some reason
                if (aggregationOperationFound)
                {
                    throw std::out_of_range("Unsupported aggregation type result");
                }
                break;
            }
        }

        if (!aggregationOperationFound)
        {
            responseMessage->mutable_payloads()->at(trimmedKey).MergeFrom(payload);
        }
    }
}

void GpuSqlDispatcher::MergePayloadToSelfResponse(const std::string& key,
                                                  ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                                  const std::string& nullBitMaskString)
{
    std::string trimmedKey = key.substr(0, std::string::npos);
    if (!key.empty() && key.front() == '$')
    {
        trimmedKey = key.substr(1, std::string::npos);
    }
    MergePayload(trimmedKey, &responseMessage_, payload);
    if (!nullBitMaskString.empty())
    {
        MergePayloadBitmask(trimmedKey, &responseMessage_, nullBitMaskString);
    }
}

bool GpuSqlDispatcher::IsRegisterAllocated(const std::string& reg)
{
    return allocatedPointers_.find(reg) != allocatedPointers_.end();
}

std::pair<std::string, std::string> GpuSqlDispatcher::SplitColumnName(const std::string& colName)
{
    const size_t splitIdx = colName.find(".");
    const std::string table = colName.substr(0, splitIdx);
    const std::string column = colName.substr(splitIdx + 1);
    return {table, column};
}

bool GpuSqlDispatcher::isValidCast(DataType fromType, DataType toType)
{
    const bool isToTypeNumeric = (toType >= COLUMN_INT && toType <= COLUMN_DOUBLE) || toType == COLUMN_INT8_T;

    if (toType == COLUMN_STRING)
    {
        return true;
    }

    else if (toType == COLUMN_POINT)
    {
        return fromType == COLUMN_STRING || toType == COLUMN_POINT;
    }

    else if (toType == COLUMN_POLYGON)
    {
        return fromType == COLUMN_STRING || toType == COLUMN_POLYGON;
    }

    else if (isToTypeNumeric)
    {
        return fromType != COLUMN_POINT && fromType != COLUMN_POLYGON;
    }

    return false;
}
