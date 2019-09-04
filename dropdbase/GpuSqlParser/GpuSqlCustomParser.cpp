#include "GpuSqlCustomParser.h"
//
// Created by Martin Staňo on 2019-01-14.
//
#include "GpuSqlParser.h"
#include "GpuSqlLexer.h"
#include "GpuSqlCustomParser.h"
#include "GpuSqlListener.h"
#include "CpuWhereListener.h"
#include "ParserExceptions.h"
#include "../QueryEngine/GPUMemoryCache.h"
#include "QueryType.h"
#include "../QueryEngine/GPUCore/IGroupBy.h"
#include "../QueryEngine/Context.h"
#include <google/protobuf/message.h>
#include "../messages/QueryResponseMessage.pb.h"
#include "../Database.h"
#include "LoadColHelper.h"
#include <iostream>
#include <future>
#include <thread>

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database>& database, const std::string& query)
: database_(database), isSingleGpuStatement_(false), query_(query)
{
    LoadColHelper& loadColHelper = LoadColHelper::getInstance();
    loadColHelper.countSkippedBlocks = 0;
}

/// Parses SQL statement
/// SELECT statment is parsed in order: FROM, WHERE, GROUP BY, SELECT, LIMIT, OFFSET, ORDER BY
/// Other statments are parsed as whole
/// SELECT statement supports multi-gpu execution
/// One dummy dispatcher instance is created to populate argument and operation queues
/// One real dispatcher instance is created for every CUDA device found on the machine which inherits executin data from dummy dispatcher
/// All real dispatcher instances are exucuted in separate threads and their results are aggregated
/// Limit anf Offset are applied on final result set
/// <returns="responseMessage">Final protobuf response message of executed statement (query result set)</returns>
DispatcherResult GpuSqlCustomParser::Parse()
{
    Context& context = Context::getInstance();
    dispatchers_.clear();
    wasAborted_ = false;
    antlr4::ANTLRInputStream sqlInputStream(query_);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    std::unique_ptr<ThrowErrorListener> throwErrorListener = std::make_unique<ThrowErrorListener>();

    sqlLexer.removeErrorListeners();
    sqlLexer.addErrorListener(throwErrorListener.get());

    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
    parser.removeErrorListeners();
    parser.addErrorListener(throwErrorListener.get());
    parser.getInterpreter<antlr4::atn::ParserATNSimulator>()->setPredictionMode(antlr4::atn::PredictionMode::SLL);

    GpuSqlParser::StatementContext* statement = parser.statement();

    antlr4::tree::ParseTreeWalker walker;

    std::vector<std::unique_ptr<IGroupBy>> groupByInstances;
    std::vector<OrderByBlocks> orderByBlocks(Context::getInstance().getDeviceCount());

    for (int i = 0; i < context.getDeviceCount(); i++)
    {
        groupByInstances.emplace_back(nullptr);
    }

    std::unique_ptr<CpuSqlDispatcher> cpuWhereDispatcher = std::make_unique<CpuSqlDispatcher>(database_);
    std::unique_ptr<GpuSqlDispatcher> dispatcher =
        std::make_unique<GpuSqlDispatcher>(database_, groupByInstances, orderByBlocks, -1);
    joinDispatcher_ = std::make_unique<GpuSqlJoinDispatcher>(database_);

    GpuSqlListener gpuSqlListener(database_, *dispatcher, *joinDispatcher_);

    CpuWhereListener cpuWhereListener(database_, *cpuWhereDispatcher);

    if (statement->sqlSelect())
    {
        if (database_ == nullptr)
        {
            throw DatabaseNotUsedException();
        }

        walker.walk(&gpuSqlListener, statement->sqlSelect()->fromTables());
        walker.walk(&cpuWhereListener, statement->sqlSelect()->fromTables());

        gpuSqlListener.ExtractColumnAliasContexts(statement->sqlSelect()->selectColumns());
        gpuSqlListener.LockAliasRegisters();
        cpuWhereListener.ExtractColumnAliasContexts(statement->sqlSelect()->selectColumns());

        if (statement->sqlSelect()->joinClauses())
        {
            walker.walk(&gpuSqlListener, statement->sqlSelect()->joinClauses());
            joinDispatcher_->Execute();
        }

        if (statement->sqlSelect()->whereClause())
        {
            walker.walk(&gpuSqlListener, statement->sqlSelect()->whereClause());
            walker.walk(&cpuWhereListener, statement->sqlSelect()->whereClause());
        }

        if (statement->sqlSelect()->groupByColumns())
        {
            walker.walk(&gpuSqlListener, statement->sqlSelect()->groupByColumns());
        }

        std::vector<GpuSqlParser::SelectColumnContext*> aggColumns;
        std::vector<GpuSqlParser::SelectColumnContext*> nonAggColumns;

        for (auto column : statement->sqlSelect()->selectColumns()->selectColumn())
        {
            if (ContainsAggregation(column))
            {
                aggColumns.push_back(column);
            }
            else
            {
                nonAggColumns.push_back(column);
            }
        }

        for (auto column : statement->sqlSelect()->selectColumns()->selectAllColumns())
        {
            gpuSqlListener.exitSelectAllColumns(column);
            break;
        }

        for (auto column : aggColumns)
        {
            walker.walk(&gpuSqlListener, column);
        }

        for (auto column : nonAggColumns)
        {
            walker.walk(&gpuSqlListener, column);
        }

        if (statement->sqlSelect()->orderByColumns())
        {
            gpuSqlListener.enterOrderByColumns(statement->sqlSelect()->orderByColumns());

            // flip the order of ORDER BY columns
            std::vector<GpuSqlParser::OrderByColumnContext*> orderByColumns;

            for (auto orderByCol : statement->sqlSelect()->orderByColumns()->orderByColumn())
            {
                orderByColumns.insert(orderByColumns.begin(), orderByCol);
            }

            for (auto orderByCol : orderByColumns)
            {
                walker.walk(&gpuSqlListener, orderByCol);
            }

            gpuSqlListener.exitOrderByColumns(statement->sqlSelect()->orderByColumns());
        }

        gpuSqlListener.exitSelectColumns(statement->sqlSelect()->selectColumns());

        if (statement->sqlSelect()->offset())
        {
            walker.walk(&gpuSqlListener, statement->sqlSelect()->offset());
        }

        if (statement->sqlSelect()->limit())
        {
            walker.walk(&gpuSqlListener, statement->sqlSelect()->limit());
        }
    }
    else if (statement->showStatement())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->showStatement());
    }
    else if (statement->sqlInsertInto())
    {
        if (database_ == nullptr)
        {
            throw DatabaseNotUsedException();
        }

        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlInsertInto());
    }
    else if (statement->sqlCreateDb())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlCreateDb());
    }
    else if (statement->sqlDropDb())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlDropDb());
    }
    else if (statement->sqlCreateTable())
    {
        if (database_ == nullptr)
        {
            throw DatabaseNotUsedException();
        }

        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlCreateTable());
    }
    else if (statement->sqlDropTable())
    {
        if (database_ == nullptr)
        {
            throw DatabaseNotUsedException();
        }

        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlDropTable());
    }
    else if (statement->sqlAlterTable())
    {
        if (database_ == nullptr)
        {
            throw DatabaseNotUsedException();
        }

        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlAlterTable());
    }
    else if (statement->sqlCreateIndex())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlCreateIndex());
    }
    if (wasAborted_)
    {
		return {};
    }
    int32_t threadCount = isSingleGpuStatement_ ? 1 : context.getDeviceCount();

    GpuSqlDispatcher::ResetGroupByCounters();
    GpuSqlDispatcher::ResetOrderByCounters();

    std::vector<std::thread> dispatcherFutures;
    std::vector<std::exception_ptr> dispatcherExceptions;
    std::vector<DispatcherResult> dispatcherResults;
    std::vector<std::string> lockList;
    if (database_)
    {
        const std::string dbName = database_->GetName();
        for (auto& tableName : GpuSqlDispatcher::linkTable)
        {
            lockList.push_back(dbName + "." + tableName.first);
        }
        GPUMemoryCache::SetLockList(lockList);
    }


    for (int i = 0; i < threadCount; i++)
    {
        dispatcherResults.emplace_back(nullptr);
        dispatcherExceptions.emplace_back(nullptr);
    }

    for (int i = 0; i < threadCount; i++)
    {
        dispatchers_.emplace_back(
            std::make_unique<GpuSqlDispatcher>(database_, groupByInstances, orderByBlocks, i));
        dispatcher->CopyExecutionDataTo(*dispatchers_[i], *cpuWhereDispatcher);
        dispatchers_[i]->SetJoinIndices(joinDispatcher_->GetJoinIndices());
        dispatcherFutures.push_back(
            std::thread(std::bind(&GpuSqlDispatcher::Execute, dispatchers_[i].get(),
                                  std::ref(dispatcherResults[i]), std::ref(dispatcherExceptions[i]))));
    }

    for (int i = 0; i < threadCount; i++)
    {
        dispatcherFutures[i].join();
        CudaLogBoost::getInstance(CudaLogBoost::info) << "TID: " << i << " Done" << '\n';
    }

    int32_t currentDev = context.getBoundDeviceID();
    for (int i = 0; i < threadCount; i++)
    {
        context.bindDeviceToContext(i);
        groupByInstances[i] = nullptr;
    }
    context.bindDeviceToContext(currentDev);

    for (int i = 0; i < threadCount; i++)
    {
        if (dispatcherExceptions[i])
        {
            std::rethrow_exception(dispatcherExceptions[i]);
        }
    }

    if (wasAborted_)
    {
		return {};
    }
    auto ret = (MergeDispatcherResults(dispatcherResults, gpuSqlListener.ResultLimit, gpuSqlListener.ResultOffset));

    return ret;
}

void GpuSqlCustomParser::InterruptQueryExecution()
{
    for (auto& dispatcher : dispatchers_)
    {
        dispatcher->Abort();
    }
    joinDispatcher_->Abort();
    wasAborted_ = true;
}

/// Merges partial dispatcher respnse messages to final response message
/// <param="dispatcherResults">Partial dispatcher result messages</param>
/// <param="resultLimit">Row limit</param>
/// <param="resultOffset">Row offset</param>
/// <returns="reponseMessage">Merged response message</returns>
DispatcherResult
GpuSqlCustomParser::MergeDispatcherResults(std::vector<DispatcherResult>& dispatcherResults,
                                           int64_t resultLimit,
                                           int64_t resultOffset)
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Limit: " << resultLimit << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Offset: " << resultOffset << '\n';

	DispatcherResult responseMessage;
    for (auto& partialResult : dispatcherResults)
    {
		responseMessage.MergeData(partialResult);
		responseMessage.MergeNullMasks(partialResult);
    }

    TrimResponseMessage(responseMessage, resultLimit, resultOffset);
    return std::move(responseMessage);
}

/// Trims all payloads of result message according to limit and offset
/// <param="responseMessage">Response message to be trimmed</param>
/// <param="limit">Row limit</param>
/// <param="offset">Row offset</param>
void GpuSqlCustomParser::TrimResponseMessage(DispatcherResult& responseMessage, int64_t limit, int64_t offset)
{
    for (auto& queryPayload : responseMessage.ResultData)
    {
        std::string key = queryPayload.first;
        TrimPayload(queryPayload.second, limit, offset);
    }
}

/// Trims single payload of result message according to limit and offset
/// <param="payload">Payload to be trimmed</param>
/// <param="nullMask">Null mask of the payload</param>
/// <param="limit">Row limit</param>
/// <param="offset">Row offset</param>
void GpuSqlCustomParser::TrimPayload(std::unique_ptr<IVariantArray>& payload,
	std::string& nullMask,
	int64_t limit,
	int64_t offset)
{
	int64_t payloadSize = payload->GetSize();
	int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
	int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);
	TrimPayload(payload, limit, offset);
	int64_t byteOffset = (clampedOffset) / (sizeof(int8_t) * 8);
	nullMask.erase(0, byteOffset);
	int32_t shiftOffset = (clampedOffset) % (sizeof(int8_t) * 8);
	int8_t carry = 0;
	nullMask[0] >>= shiftOffset;
	for (size_t i = 1; i < nullMask.size(); i++)
	{
		int8_t carry = (nullMask[i] & ((1 << shiftOffset) - 1)) << ((sizeof(int8_t) * 8) - shiftOffset);
		nullMask[i - 1] |= carry;
		nullMask[i] >>= shiftOffset;
	}
	int64_t byteLimit = (clampedLimit) / (sizeof(int8_t) * 8);
	nullMask.erase(byteLimit + 1);
}

/// Trims single payload of result message according to limit and offset
/// <param="payload">Payload to be trimmed</param>
/// <param="limit">Row limit</param>
/// <param="offset">Row offset</param>
void GpuSqlCustomParser::TrimPayload(std::unique_ptr<IVariantArray>& payload,
                                     int64_t limit,
                                     int64_t offset)
{
	int64_t payloadSize = payload->GetSize();
	int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
	int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);
    switch (payload->GetType())
    {
    case COLUMN_INT:
    {
		VariantArray<int32_t>* data = dynamic_cast<VariantArray<int32_t>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<int32_t>>(clampedLimit);
		VariantArray<int32_t>* newDataPtr = dynamic_cast<VariantArray<int32_t>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;

	case COLUMN_FLOAT:
    {
		VariantArray<float>* data = dynamic_cast<VariantArray<float>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<float>>(clampedLimit);
		VariantArray<float>* newDataPtr = dynamic_cast<VariantArray<float>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
    case COLUMN_LONG:
    {
		VariantArray<int64_t>* data = dynamic_cast<VariantArray<int64_t>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<int64_t>>(clampedLimit);
		VariantArray<int64_t>* newDataPtr = dynamic_cast<VariantArray<int64_t>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
    case COLUMN_DOUBLE:
    {
		VariantArray<double>* data = dynamic_cast<VariantArray<double>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<double>>(clampedLimit);
		VariantArray<double>* newDataPtr = dynamic_cast<VariantArray<double>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
    case COLUMN_POINT:
    {
		VariantArray<ColmnarDB::Types::Point>* data = dynamic_cast<VariantArray<ColmnarDB::Types::Point>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<ColmnarDB::Types::Point>>(clampedLimit);
		VariantArray<ColmnarDB::Types::Point>* newDataPtr = dynamic_cast<VariantArray<ColmnarDB::Types::Point>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
    case COLUMN_POLYGON:
    {
		VariantArray<ColmnarDB::Types::ComplexPolygon>* data = dynamic_cast<VariantArray<ColmnarDB::Types::ComplexPolygon>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<ColmnarDB::Types::ComplexPolygon>>(clampedLimit);
		VariantArray<ColmnarDB::Types::ComplexPolygon>* newDataPtr = dynamic_cast<VariantArray<ColmnarDB::Types::ComplexPolygon>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
    case COLUMN_STRING:
	{
		VariantArray<std::string>* data = dynamic_cast<VariantArray<std::string>*>(payload.get());
		std::unique_ptr<IVariantArray> newData = std::make_unique <VariantArray<std::string>>(clampedLimit);
		VariantArray<std::string>* newDataPtr = dynamic_cast<VariantArray<std::string>*>(newData.get());
		std::copy(data->getData() + clampedOffset, data->getData() + clampedOffset + clampedLimit, newDataPtr->getData());
		payload.swap(newData);
    }
    break;
	default:
		break;
    }
}

bool GpuSqlCustomParser::ContainsAggregation(GpuSqlParser::SelectColumnContext* ctx)
{
    antlr4::tree::ParseTreeWalker walker;

    class : public GpuSqlParserBaseListener
    {
    public:
        bool containsAggregation = false;

    private:
        void exitAggregation(GpuSqlParser::AggregationContext* ctx) override
        {
            containsAggregation = true;
        }

    } findAggListener;

    walker.walk(&findAggListener, ctx);

    return findAggListener.containsAggregation;
}

void ThrowErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                     antlr4::Token* offendingSymbol,
                                     size_t line,
                                     size_t charPositionInLine,
                                     const std::string& msg,
                                     std::exception_ptr e)
{
    std::string finalMsg =
        "Error : line " + std::to_string(line) + ":" + std::to_string(charPositionInLine) + " " + msg;
    throw antlr4::ParseCancellationException(finalMsg);
}
