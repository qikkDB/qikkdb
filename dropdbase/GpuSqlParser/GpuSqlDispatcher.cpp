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
#include "../ConstraintType.h"
#include "LoadColHelper.h"
#include "InsertIntoStruct.h"
#include <any>
#include <string>
#include <unordered_map>
#include <boost/filesystem.hpp>

const std::string GpuSqlDispatcher::KEYS_SUFFIX = "_keys";
const std::string GpuSqlDispatcher::NULL_SUFFIX = "_nullMask";
const std::string GpuSqlDispatcher::RECONSTRUCTED_SUFFIX = "_reconstructed";

int32_t GpuSqlDispatcher::groupByDoneCounter_ = 0;
int32_t GpuSqlDispatcher::orderByDoneCounter_ = 0;
int64_t GpuSqlDispatcher::processedDataSize_ = 0;

std::mutex GpuSqlDispatcher::groupByMutex_;
std::mutex GpuSqlDispatcher::orderByMutex_;
std::mutex GpuSqlDispatcher::loadSizeMutex_;

std::atomic_bool GpuSqlDispatcher::thrownException_(false);

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
  groupByHashTableFull_(false), hashTableMultiplier_(1), loadNecessary_(1), cpuDispatcher_(database),
  jmpInstructionPosition_(0), insertIntoData_(std::make_unique<InsertIntoStruct>()),
  joinIndices_(nullptr), orderByTable_(nullptr), orderByBlocks_(orderByBlocks),
  loadedTableName_(""), loadSize_(0), loadOffset_(0)
{
}

GpuSqlDispatcher::~GpuSqlDispatcher()
{
}

void GpuSqlDispatcher::SetLoadedTableName(const std::string& tableName)
{
    loadedTableName_ = tableName;
}

void GpuSqlDispatcher::AddGetLoadSizeFunction()
{
    dispatcherFunctions_.push_back(getLoadSizeFunction_);
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

        InstructionStatus err = InstructionStatus::CONTINUE;

        while (err == InstructionStatus::CONTINUE && !aborted_ && !thrownException_)
        {

            err = (this->*dispatcherFunctions_[instructionPointer_++])();
#ifndef NDEBUG
            printf("tid:%d ip: %d \n", dispatcherThreadId_, instructionPointer_ - 1);
            AssertDeviceMatchesCurrentThread(dispatcherThreadId_);
#endif
            // Print logs
            if (err != InstructionStatus::CONTINUE)
            {
                switch (err)
                {
                case InstructionStatus::OUT_OF_BLOCKS:
                    CudaLogBoost::getInstance(CudaLogBoost::info) << "Out of blocks" << '\n';
                    break;
                case InstructionStatus::FINISH:
                    // do nothing, logs are in appropriate instructions
                    break;
                case InstructionStatus::LOAD_SKIPPED:
                    CudaLogBoost::getInstance(CudaLogBoost::info) << "Load skipped" << '\n';
                    loadColHelper.countSkippedBlocks++;
                    err = InstructionStatus::CONTINUE;
                    break;
                case InstructionStatus::EXCEPTION:
                    CudaLogBoost::getInstance(CudaLogBoost::error)
                        << "Abort Dispatch Execution, exception thrown in some thread"
                        << "\n";
                    break;
                }
            }
            // Check err again because for LOAD_SKIPPED case it was changed
            if (err != InstructionStatus::CONTINUE)
            {
                break; // Stop execution
            }
        }
        result = std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>(
            std::move(responseMessage_));
    }
    catch (...)
    {
        exception = std::current_exception();
        GpuSqlDispatcher::thrownException_ = true;
        GpuSqlDispatcher::groupByCV_.notify_all();
        GpuSqlDispatcher::orderByCV_.notify_all();
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
    dispatcherFunctions_.push_back(retFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddOrderByFunction(DataType type)
{
    dispatcherFunctions_.push_back(orderByFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddOrderByReconstructFunction(DataType type)
{
    dispatcherFunctions_.push_back(orderByReconstructFunctions_[GetUnaryDispatchTableIndex(type)]);
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

void GpuSqlDispatcher::AddShowConstraintsFunction()
{
    dispatcherFunctions_.push_back(showConstraintsFunction_);
}

void GpuSqlDispatcher::AddShowQueryColumnTypesFunction()
{
    dispatcherFunctions_.push_back(showQueryColumnTypesFunction_);
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

void GpuSqlDispatcher::AddAlterDatabaseFunction()
{
    dispatcherFunctions_.push_back(alterDatabaseFunction_);
}

void GpuSqlDispatcher::AddCreateIndexFunction()
{
    dispatcherFunctions_.push_back(createIndexFunction_);
}

void GpuSqlDispatcher::AddInsertIntoFunction(DataType type)
{
    dispatcherFunctions_.push_back(insertIntoFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddInsertIntoDoneFunction()
{
    dispatcherFunctions_.push_back(insertIntoDoneFunction_);
}

int32_t GpuSqlDispatcher::GetBinaryDispatchTableIndex(DataType left, DataType right)
{
    DataType lConst = GetConstDataType(left);
    DataType rConst = GetConstDataType(right);
    int32_t constOffset = 0;
    if (left >= numOfDataTypes && right >= numOfDataTypes)
    {
        constOffset = 3;
    }
    else if (left >= numOfDataTypes)
    {
        constOffset = 2;
    }
    else if (right >= numOfDataTypes)
    {
        constOffset = 1;
    }
    return numOfDataTypes * 4 * lConst + 4 * rConst + constOffset;
}

int32_t GpuSqlDispatcher::GetUnaryDispatchTableIndex(DataType type)
{
    DataType lConst = GetConstDataType(type);
    int32_t constOffset = 0;
    if (type >= numOfDataTypes)
    {
        constOffset = 1;
    }
    return 2 * lConst + constOffset;
}

void GpuSqlDispatcher::ResetBlocksProcessing()
{
    CudaLogBoost::getInstance(CudaLogBoost::info)
        << "Restart blocks processing in thread " << dispatcherThreadId_ << '\n';
    CleanUpGpuPointers();
    blockIndex_ = dispatcherThreadId_;
    instructionPointer_ = 0;
    insideAggregation_ = false;
    insideGroupBy_ = false;
    usingGroupBy_ = false;
    usingOrderBy_ = false;
    usingJoin_ = false;
    isLastBlockOfDevice_ = false;
    isOverallLastBlock_ = false;
    noLoad_ = true;
    aborted_ = false;
}

void GpuSqlDispatcher::HandleHashTableFull()
{
    // Set flags to restart a blocks processing by current thread
    instructionPointer_ = jmpInstructionPosition_;
    groupByHashTableFull_ = true;
    groupByTables_[dispatcherThreadId_].reset();

    // Heuristic for new needed hash table buffers size. Approximate
    // based on progress of overall blocks processing (block count / block index)
    const int32_t divider = blockIndex_ - dispatcherThreadId_ + Context::getInstance().getDeviceCount();
    hashTableMultiplier_ *= (GetBlockCount() + divider / 2) / divider + 1;

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Increased hash table size to "
        << std::min(static_cast<size_t>(Configuration::GetInstance().GetGroupByBuckets()) * hashTableMultiplier_,
                    GB_BUFFER_SIZE_MAX)
        << '\n';
}

void GpuSqlDispatcher::AddGreaterFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(greaterFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddLessFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(lessFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddGreaterEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(greaterEqualFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddLessEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(lessEqualFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(equalFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddNotEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(notEqualFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddLogicalAndFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(logicalAndFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddLogicalOrFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(logicalOrFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddMulFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(mulFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
}


void GpuSqlDispatcher::AddDivFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(divFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddAddFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(addFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}


void GpuSqlDispatcher::AddSubFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(subFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddModFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(modFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddBitwiseOrFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseOrFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddBitwiseAndFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseAndFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddBitwiseXorFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseXorFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddBitwiseLeftShiftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseLeftShiftFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddBitwiseRightShiftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(bitwiseRightShiftFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddPointFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(pointFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddLogarithmFunction(DataType number, DataType base)
{
    dispatcherFunctions_.push_back(logarithmFunctions_[GetBinaryDispatchTableIndex(number, base)]);
}

void GpuSqlDispatcher::AddArctangent2Function(DataType y, DataType x)
{
    dispatcherFunctions_.push_back(arctangent2Functions_[GetBinaryDispatchTableIndex(y, x)]);
}

void GpuSqlDispatcher::AddRoundDecimalFunction(DataType y, DataType x)
{
    dispatcherFunctions_.push_back(roundDecimalFunctions_[GetBinaryDispatchTableIndex(y, x)]);
}

void GpuSqlDispatcher::AddConcatFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(concatFunctions[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddLeftFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(leftFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddRightFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(rightFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddPowerFunction(DataType base, DataType exponent)
{
    dispatcherFunctions_.push_back(powerFunctions_[GetBinaryDispatchTableIndex(base, exponent)]);
}

void GpuSqlDispatcher::AddRootFunction(DataType base, DataType exponent)
{
    dispatcherFunctions_.push_back(rootFunctions_[GetBinaryDispatchTableIndex(base, exponent)]);
}

void GpuSqlDispatcher::AddGeoLongitudeToTileXFunction(DataType longitude, DataType zoom)
{
    dispatcherFunctions_.push_back(geoLongitudeToTileXFunctions_[GetBinaryDispatchTableIndex(longitude, zoom)]);
}

void GpuSqlDispatcher::AddGeoLatitudeToTileYFunction(DataType latitude, DataType zoom)
{
    dispatcherFunctions_.push_back(geoLatitudeToTileYFunctions_[GetBinaryDispatchTableIndex(latitude, zoom)]);
}

void GpuSqlDispatcher::AddGeoTileXToLongitudeFunction(DataType tileX, DataType zoom)
{
    dispatcherFunctions_.push_back(geoTileXToLongitudeFunctions_[GetBinaryDispatchTableIndex(tileX, zoom)]);
}

void GpuSqlDispatcher::AddGeoTileYToLatitudeFunction(DataType tileY, DataType zoom)
{
    dispatcherFunctions_.push_back(geoTileYToLatitudeFunctions_[GetBinaryDispatchTableIndex(tileY, zoom)]);
}

void GpuSqlDispatcher::AddContainsFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(containsFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddIntersectFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(intersectFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddUnionFunction(DataType left, DataType right)
{
    dispatcherFunctions_.push_back(unionFunctions_[GetBinaryDispatchTableIndex(left, right)]);
}

void GpuSqlDispatcher::AddCastToIntFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToIntFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToLongFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToLongFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToDateFunction(DataType operand)
{
    // dispatcherFunctions_.push_back(castToDateFunctions[operand]);
}

void GpuSqlDispatcher::AddCastToFloatFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToFloatFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToDoubleFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToDoubleFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToStringFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToStringFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToPointFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToPointFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToPolygonFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToPolygonFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddCastToInt8TFunction(DataType operand)
{
    dispatcherFunctions_.push_back(castToInt8TFunctions_[GetUnaryDispatchTableIndex(operand)]);
}

void GpuSqlDispatcher::AddLogicalNotFunction(DataType type)
{
    dispatcherFunctions_.push_back(logicalNotFunctions_[GetUnaryDispatchTableIndex(type)]);
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
    dispatcherFunctions_.push_back(minusFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddDateToStringFunction(DataType type)
{
    dispatcherFunctions_.push_back(dateToStringFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddYearFunction(DataType type)
{
    dispatcherFunctions_.push_back(yearFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddMonthFunction(DataType type)
{
    dispatcherFunctions_.push_back(monthFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddDayFunction(DataType type)
{
    dispatcherFunctions_.push_back(dayFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddHourFunction(DataType type)
{
    dispatcherFunctions_.push_back(hourFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddMinuteFunction(DataType type)
{
    dispatcherFunctions_.push_back(minuteFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddSecondFunction(DataType type)
{
    dispatcherFunctions_.push_back(secondFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddAbsoluteFunction(DataType type)
{
    dispatcherFunctions_.push_back(absoluteFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddSineFunction(DataType type)
{
    dispatcherFunctions_.push_back(sineFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddCosineFunction(DataType type)
{
    dispatcherFunctions_.push_back(cosineFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddTangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(tangentFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddCotangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(cotangentFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddArcsineFunction(DataType type)
{
    dispatcherFunctions_.push_back(arcsineFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddArccosineFunction(DataType type)
{
    dispatcherFunctions_.push_back(arccosineFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddArctangentFunction(DataType type)
{
    dispatcherFunctions_.push_back(arctangentFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddLogarithm10Function(DataType type)
{
    dispatcherFunctions_.push_back(logarithm10Functions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddLogarithmNaturalFunction(DataType type)
{
    dispatcherFunctions_.push_back(logarithmNaturalFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddExponentialFunction(DataType type)
{
    dispatcherFunctions_.push_back(exponentialFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddSquareFunction(DataType type)
{
    dispatcherFunctions_.push_back(squareFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddSquareRootFunction(DataType type)
{
    dispatcherFunctions_.push_back(squareRootFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddSignFunction(DataType type)
{
    dispatcherFunctions_.push_back(signFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddRoundFunction(DataType type)
{
    dispatcherFunctions_.push_back(roundFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddFloorFunction(DataType type)
{
    dispatcherFunctions_.push_back(floorFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddCeilFunction(DataType type)
{
    dispatcherFunctions_.push_back(ceilFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddLtrimFunction(DataType type)
{
    dispatcherFunctions_.push_back(ltrimFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddRtrimFunction(DataType type)
{
    dispatcherFunctions_.push_back(rtrimFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddLowerFunction(DataType type)
{
    dispatcherFunctions_.push_back(lowerFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddUpperFunction(DataType type)
{
    dispatcherFunctions_.push_back(upperFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddReverseFunction(DataType type)
{
    dispatcherFunctions_.push_back(reverseFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddLenFunction(DataType type)
{
    dispatcherFunctions_.push_back(lenFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddMinFunction(DataType key, DataType value, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::NO_GROUP_BY:
        fun = minAggregationFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = minGroupByFunctions_[GetBinaryDispatchTableIndex(key, value)];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = minGroupByMultiKeyFunctions_[GetUnaryDispatchTableIndex(value)];
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
        fun = maxAggregationFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = maxGroupByFunctions_[GetBinaryDispatchTableIndex(key, value)];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = maxGroupByMultiKeyFunctions_[GetUnaryDispatchTableIndex(value)];
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
        fun = sumAggregationFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = sumGroupByFunctions_[GetBinaryDispatchTableIndex(key, value)];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = sumGroupByMultiKeyFunctions_[GetUnaryDispatchTableIndex(value)];
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
        fun = countAggregationFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = countGroupByFunctions_[GetBinaryDispatchTableIndex(key, value)];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = countGroupByMultiKeyFunctions_[GetUnaryDispatchTableIndex(value)];
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
        fun = avgAggregationFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = avgGroupByFunctions_[GetBinaryDispatchTableIndex(key, value)];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = avgGroupByMultiKeyFunctions_[GetUnaryDispatchTableIndex(value)];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
}

void GpuSqlDispatcher::AddGroupByFunction(DataType type)
{
    dispatcherFunctions_.push_back(groupByFunctions_[GetUnaryDispatchTableIndex(type)]);
}

void GpuSqlDispatcher::AddGroupByBeginFunction()
{
    dispatcherFunctions_.push_back(groupByBeginFunction_);
}

void GpuSqlDispatcher::AddGroupByDoneFunction(DataType key, GroupByType groupByType)
{
    GpuSqlDispatcher::DispatchFunction fun;
    switch (groupByType)
    {
    case GroupByType::SINGLE_KEY_GROUP_BY:
        fun = groupByDoneFunctions_[key];
        break;
    case GroupByType::MULTI_KEY_GROUP_BY:
        fun = groupByDoneFunctions_[DataType::DATA_TYPE_SIZE];
        break;
    default:
        break;
    }
    dispatcherFunctions_.push_back(fun);
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

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::LoadColNullMask(std::string& colName)
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

        auto blockNullMask =
            database_->GetTables().at(table).GetColumns().at(column)->GetNullBitMaskForBlock(blockIndex_);
        size_t blockNullMaskSize = (loadSize_ + 8 * sizeof(int8_t) - 1) / (8 * sizeof(int8_t));

        auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
            database_->GetName(), colName + NULL_SUFFIX, blockIndex_, blockNullMaskSize, loadSize_, loadOffset_);
        if (!std::get<2>(cacheEntry))
        {
            GPUMemory::copyHostToDevice(std::get<0>(cacheEntry), std::get<0>(blockNullMask), blockNullMaskSize);
        }
        AddCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheEntry), loadSize_);

        noLoad_ = false;
    }
    return InstructionStatus::CONTINUE;
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

            FillPolygonRegister(polygon, colName, size, useCache, nullMaskPtr);

            return polygon;
        }
        else
        {
            GPUMemory::GPUPolygon polygon =
                ComplexPolygonFactory::PrepareGPUPolygon(polygons, databaseName, colName,
                                                         blockIndex_, loadSize_, loadOffset_);
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
                                                    const std::string* strings,
                                                    const size_t stringCount,
                                                    bool useCache,
                                                    int8_t* nullMaskPtr)
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
            FillStringRegister(gpuString, colName, stringCount, useCache, nullMaskPtr);
            return gpuString;
        }
        else
        {
            GPUMemory::GPUString gpuString =
                StringFactory::PrepareGPUString(strings, stringCount, databaseName, colName,
                                                blockIndex_, loadSize_, loadOffset_);
            FillStringRegister(gpuString, colName, stringCount, useCache, nullMaskPtr);
            return gpuString;
        }
    }
    else
    {
        GPUMemory::GPUString gpuString = StringFactory::PrepareGPUString(strings, stringCount);
        FillStringRegister(gpuString, colName, stringCount, useCache, nullMaskPtr);
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

GPUMemory::GPUString GpuSqlDispatcher::InsertConstStringGpu(const std::string& str, const size_t size)
{
    std::vector<std::string> strings(size, str);
    std::string name = "constString" + std::to_string(constStringCounter_);
    constStringCounter_++;
    return InsertString(database_->GetName(), name, strings.data(), size);
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
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Fil()
{
    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Filter: " << reg << '\n';
    filter_ = allocatedPointers_.at(reg).GpuPtr;
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::WhereEvaluation()
{
    bool containsAggFunction = arguments_.Read<bool>();
    // loadNecessary_ = (usingJoin_ || containsAggFunction) ? 1 : cpuDispatcher_.Execute(blockIndex_);
    loadNecessary_ = usingJoin_ ? 1 : cpuDispatcher_.Execute(blockIndex_);
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Where load evaluation: " << loadNecessary_ << '\n';
    return InstructionStatus::CONTINUE;
}


/// Implementation of JMP operation
/// Determines next block index to process by this instance of dispatcher based on CUDA device count
/// <returns name="statusCode">Finish status code of the operation</returns>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Jmp()
{
    Context& context = Context::getInstance();

    // unlock GB and OB mutexes if load was skipped and block was last on device
    if (noLoad_ == false && loadNecessary_ == 0 && isLastBlockOfDevice_)
    {
        {
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::orderByMutex_);
            GpuSqlDispatcher::IncOrderByDoneCounter();
            GpuSqlDispatcher::orderByCV_.notify_all();
        }
        {
            std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
            GpuSqlDispatcher::IncGroupByDoneCounter();
            GpuSqlDispatcher::groupByCV_.notify_all();
        }
    }

    if (noLoad_ && loadNecessary_ != 0)
    {
        CleanUpGpuPointers();
        return InstructionStatus::CONTINUE;
    }

    if (groupByHashTableFull_)
    {
        groupByHashTableFull_ = false;
        ResetBlocksProcessing();
        context.getCacheForCurrentDevice().setCurrentBlockIndex(blockIndex_);
        return InstructionStatus::CONTINUE;
    }

    if (!isLastBlockOfDevice_)
    {
        blockIndex_ += context.getDeviceCount();
        context.getCacheForCurrentDevice().setCurrentBlockIndex(blockIndex_);
        instructionPointer_ = 0;
        CleanUpGpuPointers();
        return InstructionStatus::CONTINUE;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Jump" << '\n';
    return InstructionStatus::CONTINUE;
}


/// Implementation of DONE operation
/// Clears all allocated temporary result buffers
/// <returns name="statusCode">Finish status code of the operation</returns>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Done()
{
    CleanUpGpuPointers();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Done" << '\n';
    return InstructionStatus::OUT_OF_BLOCKS;
}

/// Implementation of SHOW DATABASES operation
/// Inserts database names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ShowDatabases()
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
    MergePayloadToSelfResponse("Databases", "Databases", payload);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Show databases completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}


/// Implementation of SHOW TABLES operation
/// Inserts table names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ShowTables()
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
    MergePayloadToSelfResponse(db, db, payload);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Show tables completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

/// Implementation of SHOW COLUMN operation
/// Inserts column names and their types to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ShowColumns()
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
    MergePayloadToSelfResponse(tab + "_columns", tab + "_columns", payloadName);
    MergePayloadToSelfResponse(tab + "_types", tab + "_types", payloadType);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Show columns completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ShowConstraints()
{
    std::string db = arguments_.Read<std::string>();
    std::string tab = arguments_.Read<std::string>();

    std::shared_ptr<Database> database = Database::GetDatabaseByName(db);
    auto& table = database_->GetTables().at(tab);

    auto& constraints = table.GetConstraints();

    std::unique_ptr<std::string[]> outDataConstraintName(new std::string[constraints.size()]);
    std::unique_ptr<std::string[]> outDataConstraintType(new std::string[constraints.size()]);
    std::unique_ptr<std::string[]> outDataConstraintColumns(new std::string[constraints.size()]);

    int i = 0;
    for (auto& constraint : constraints)
    {
        outDataConstraintName[i] = constraint.first;
        outDataConstraintType[i] = ::GetConstraintTypeName(constraint.second.first);

        for (auto& constraintColumn : constraint.second.second)
        {
            outDataConstraintColumns[i] += (constraintColumn + '\n');
        }
        outDataConstraintColumns[i].pop_back();

        i++;
    }

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadConstraintName;
    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadConstraintType;
    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadConstraintColumns;

    InsertIntoPayload(payloadConstraintName, outDataConstraintName, constraints.size());
    InsertIntoPayload(payloadConstraintType, outDataConstraintType, constraints.size());
    InsertIntoPayload(payloadConstraintColumns, outDataConstraintColumns, constraints.size());

    MergePayloadToSelfResponse(tab + "_constraints", tab + "_constraints", payloadConstraintName);
    MergePayloadToSelfResponse(tab + "_cnstrn_types", tab + "_cnstrn_types", payloadConstraintType);
    MergePayloadToSelfResponse(tab + "_cnstrn_cols", tab + "_cnstrn_cols", payloadConstraintColumns);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Show constraints completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ShowQueryColumnTypes()
{
    int32_t columnSize = arguments_.Read<int32_t>();

    std::unique_ptr<std::string[]> outDataColumnName(new std::string[columnSize]);
    std::unique_ptr<std::string[]> outDataColumnType(new std::string[columnSize]);

    for (int32_t i = 0; i < columnSize; i++)
    {
        outDataColumnName[i] = arguments_.Read<std::string>();
        outDataColumnType[i] = arguments_.Read<std::string>();
    }

    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadColumnName;
    ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadColumnType;

    InsertIntoPayload(payloadColumnName, outDataColumnName, columnSize);
    InsertIntoPayload(payloadColumnType, outDataColumnType, columnSize);

    MergePayloadToSelfResponse("ColumnName", "ColumnName", payloadColumnName);
    MergePayloadToSelfResponse("TypeName", "TypeName", payloadColumnType);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Show query column types completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}


GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CreateDatabase()
{
    std::string newDbName = arguments_.Read<std::string>();
    int32_t newDbBlockSize = arguments_.Read<int32_t>();
    std::shared_ptr<Database> newDb = std::make_shared<Database>(newDbName.c_str(), newDbBlockSize);
    Database::AddToInMemoryDatabaseList(newDb);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Create database_ completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DropDatabase()
{
    std::string dbName = arguments_.Read<std::string>();
    Database::GetDatabaseByName(dbName)->DeleteDatabaseFromDisk();
    Database::RemoveFromInMemoryDatabaseList(dbName.c_str());

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Drop database_ completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CreateTable()
{
    std::unordered_map<std::string, DataType> newColumns;
    std::vector<std::tuple<std::string, ConstraintType, std::vector<std::string>>> newConstraints;

    std::string newTableName = arguments_.Read<std::string>();
    int32_t newTableBlockSize = arguments_.Read<int32_t>();

    int32_t newColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < newColumnsCount; i++)
    {
        std::string newColumnName = arguments_.Read<std::string>();
        int32_t newColumnDataType = arguments_.Read<int32_t>();

        newColumns.insert({newColumnName, static_cast<DataType>(newColumnDataType)});
    }

    int32_t newConstraintCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < newConstraintCount; i++)
    {
        std::string constraintName = arguments_.Read<std::string>();
        ConstraintType constraintType = static_cast<ConstraintType>(arguments_.Read<std::int32_t>());

        int32_t constraintColumnCount = arguments_.Read<int32_t>();
        std::vector<std::string> constraintColumns;

        for (int32_t j = 0; j < constraintColumnCount; j++)
        {
            std::string newConstraintColumn = arguments_.Read<std::string>();
            constraintColumns.push_back(newConstraintColumn);
        }
        newConstraints.push_back({constraintName, constraintType, constraintColumns});
    }

    try
    {
        database_->CreateTable(newColumns, newTableName.c_str(), std::unordered_map<std::string, bool>(),
                               std::unordered_map<std::string, bool>(), newTableBlockSize);
        for (auto& constraint : newConstraints)
        {
            database_->GetTables()
                .at(newTableName)
                .AddConstraint(std::get<0>(constraint), std::get<1>(constraint), std::get<2>(constraint));
        }
    }
    catch (const constraint_violation_error& e)
    {
        database_->GetTables().erase(newTableName.c_str());
        database_->DeleteTableFromDisk(newTableName.c_str());
        throw;
    }
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Create table completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DropTable()
{
    std::string tableName = arguments_.Read<std::string>();
    database_->GetTables().erase(tableName);
    database_->DeleteTableFromDisk(tableName.c_str());

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Drop table completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::AlterTable()
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

    int32_t renameColumnsCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < renameColumnsCount; i++)
    {
        // rename column in memory:
        std::string renameColumnNameFrom = arguments_.Read<std::string>();
        std::string renameColumnNameTo = arguments_.Read<std::string>();
        database_->GetTables().at(tableName).RenameColumn(renameColumnNameFrom, renameColumnNameTo);

        // rename column on disk, if the database was persisted already:
        auto& path = Configuration::GetInstance().GetDatabaseDir();
        if (boost::filesystem::remove(path + database_->GetName() + ".db"))
        {
            std::string prefix(path + database_->GetName() + Database::SEPARATOR + tableName + Database::SEPARATOR);
            const boost::filesystem::path& oldPath{prefix + renameColumnNameFrom + ".col"};
            const boost::filesystem::path& newPath{prefix + renameColumnNameTo + ".col"};
            boost::filesystem::rename(oldPath, newPath);

            // update changes in .db file:
            database_->PersistOnlyDbFile(path.c_str());
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Renaming column: Main (.db) file of db " << database_->GetName()
                << " was NOT removed from disk. No such file (if the database was not yet saved, "
                   "ignore this warning) or no write access.";
        }
    }

    bool tableRenamed = arguments_.Read<bool>();
    if (tableRenamed)
    {
        std::string newTableName = arguments_.Read<std::string>();
        database_->RenameTable(tableName, newTableName);
        tableName = newTableName;
    }

    int32_t newConstraintCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < newConstraintCount; i++)
    {
        std::string constraintName = arguments_.Read<std::string>();
        ConstraintType constraintType = static_cast<ConstraintType>(arguments_.Read<std::int32_t>());

        int32_t constraintColumnCount = arguments_.Read<int32_t>();
        std::vector<std::string> constraintColumns;

        for (int32_t j = 0; j < constraintColumnCount; j++)
        {
            std::string newConstraintColumn = arguments_.Read<std::string>();
            constraintColumns.push_back(newConstraintColumn);
        }
        database_->GetTables().at(tableName).AddConstraint(constraintName, constraintType, constraintColumns);
    }

    int32_t dropConstraintCount = arguments_.Read<int32_t>();
    for (int32_t i = 0; i < dropConstraintCount; i++)
    {
        std::string dropConstraintName = arguments_.Read<std::string>();
        database_->GetTables().at(tableName).DropConstraint(dropConstraintName);
    }

    int32_t newBlockSize = arguments_.Read<int32_t>();
    database_->ChangeTableBlockSize(tableName.c_str(), newBlockSize);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Alter table completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::AlterDatabase()
{
    std::string databaseName = arguments_.Read<std::string>();

    bool databaseRenamed = arguments_.Read<bool>();
    if (databaseRenamed)
    {
        // change database name in memory:
        std::string newDatabaseName = arguments_.Read<std::string>();
        Database::GetDatabaseByName(databaseName)->SetName(newDatabaseName);
        auto& loadedDatabases = Context::getInstance().GetLoadedDatabases();

        auto handler = loadedDatabases.extract(databaseName);
        handler.key() = newDatabaseName;
        loadedDatabases.insert(std::move(handler));

        // updates saved files on disk, if there are any
        auto& path = Configuration::GetInstance().GetDatabaseDir();

        // delete main .db file, persist a new .db file and rename .col files:
        if (boost::filesystem::remove(path + databaseName + ".db"))
        {
            BOOST_LOG_TRIVIAL(info) << "Renaming database: Main (.db) file of db " << databaseName
                                    << " was successfully removed from disk.";
            // persist updated .db file
            Database::GetDatabaseByName(newDatabaseName)->PersistOnlyDbFile(path.c_str());

            std::string prefix(databaseName + Database::SEPARATOR);
            std::string prefix2(path + databaseName + Database::SEPARATOR);
            for (auto& p : boost::filesystem::directory_iterator(path))
            {
                // rename .col files:
                if (!p.path().string().compare(path.size(), prefix.size(), prefix))
                {
                    std::string sufix = p.path().string().substr(prefix2.size());
                    const boost::filesystem::path& newPath{path + newDatabaseName + Database::SEPARATOR + sufix};
                    boost::filesystem::rename(p.path(), newPath);
                }
            }
            databaseName = newDatabaseName;
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning)
                << "Renaming database: Main (.db) file of db " << databaseName
                << " was NOT removed from disk. No such file (if the database was not yet saved, "
                   "ignore this warning) or no write access.";
        }
    }

    bool blockSizeChanged = arguments_.Read<bool>();
    if (blockSizeChanged)
    {
        int32_t newBlockSize = arguments_.Read<int32_t>();
        Database::GetDatabaseByName(databaseName)->ChangeDatabaseBlockSize(newBlockSize);
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Alter database completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CreateIndex()
{
    std::string indexName = arguments_.Read<std::string>();
    std::string tableName = arguments_.Read<std::string>();
    std::vector<std::string> sortingColumns;

    int32_t indexColumnCount = arguments_.Read<int32_t>();
    for (int i = 0; i < indexColumnCount; i++)
    {
        std::string indexColumn = arguments_.Read<std::string>();
        sortingColumns.push_back(indexColumn);
    }

    database_->GetTables().at(tableName).AddConstraint(indexName, ConstraintType::CONSTRAINT_INDEX, sortingColumns);

    CudaLogBoost::getInstance(CudaLogBoost::info) << "Create index completed sucessfully" << '\n';
    return InstructionStatus::FINISH;
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
        size_t dataLength = 0;
        switch (responseMessage->payloads().at(key).payload_case())
        {
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
            dataLength = responseMessage->payloads().at(key).intpayload().intdata_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
            dataLength = responseMessage->payloads().at(key).int64payload().int64data_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
            dataLength = responseMessage->payloads().at(key).floatpayload().floatdata_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
            dataLength = responseMessage->payloads().at(key).doublepayload().doubledata_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
            dataLength = responseMessage->payloads().at(key).stringpayload().stringdata_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
            dataLength = responseMessage->payloads().at(key).pointpayload().pointdata_size();
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
            dataLength = responseMessage->payloads().at(key).polygonpayload().polygondata_size();
            break;
        default:
            break;
        }
        if (dataLength % 8 == 0)
        {
            responseMessage->mutable_nullbitmasks()->at(key) += nullMask;
        }
        else
        {
            int shiftCount = 8 - (dataLength % 8);
            std::vector<int8_t> nullMaskVec(nullMask.begin(), nullMask.end());
            int8_t carryBits = nullMaskVec[0] & ((1 << shiftCount) - 1);
            responseMessage->mutable_nullbitmasks()->at(key).back() |= (carryBits << (8 - shiftCount));
            ShiftNullMaskLeft(nullMaskVec, shiftCount);
            std::string newNullMask(nullMaskVec.begin(), nullMaskVec.end());
            responseMessage->mutable_nullbitmasks()->at(key) += newNullMask;
        }
    }
}

void GpuSqlDispatcher::MergePayload(const std::string& trimmedKey,
                                    const std::string& trimmedRealName,
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

        size_t keyParensIndex = trimmedRealName.find('(');

        bool aggregationOperationFound = false;
        // If no function is used
        if (keyParensIndex == std::string::npos)
        {
            aggregationOperationFound = false;
        }
        else
        {
            // Get operation name
            std::string operation = trimmedRealName.substr(0, keyParensIndex);
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
                                                  const std::string& realName,
                                                  ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload,
                                                  const std::string& nullBitMaskString)
{
    std::string trimmedKey = key;
    std::string realTrimmedName = realName;
    if (!key.empty() && key.front() == '$')
    {
        trimmedKey = key.substr(1, std::string::npos);
    }

    if (!realName.empty() && realName.front() == '$')
    {
        realTrimmedName = realName.substr(1, std::string::npos);
    }

    if (!nullBitMaskString.empty())
    {
        MergePayloadBitmask(trimmedKey, &responseMessage_, nullBitMaskString);
    }
    MergePayload(trimmedKey, realTrimmedName, &responseMessage_, payload);
}

bool GpuSqlDispatcher::IsRegisterAllocated(const std::string& reg)
{
    return (allocatedPointers_.find(reg) != allocatedPointers_.end()) ||
           (allocatedPointers_.find(reg + "_stringIndices") != allocatedPointers_.end() &&
            allocatedPointers_.find(reg + "_allChars") != allocatedPointers_.end()) ||
           (allocatedPointers_.find(reg + "_polyPoints") != allocatedPointers_.end() &&
            allocatedPointers_.find(reg + "_pointIdx") != allocatedPointers_.end() &&
            allocatedPointers_.find(reg + "_polyIdx") != allocatedPointers_.end());
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
