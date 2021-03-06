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
#include "GpuSqlParser.h"
#include "GpuSqlLexer.h"
#include <iostream>
#include <future>
#include <thread>

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database>& database, const std::string& query)
: database_(database), isSingleGpuStatement_(false), query_(query), wasAborted_(false)
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
std::unique_ptr<google::protobuf::Message> GpuSqlCustomParser::Parse()
{
    Context& context = Context::getInstance();
    dispatchers_.clear();
    // JV: HACK HACK HACK
    // Unified dispatcher would make this nicer
    GpuSqlDispatcher::ResetErrorState();
    wasAborted_ = false;
    antlr4::ANTLRInputStream sqlInputStream(query_);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    std::unique_ptr<ThrowErrorListener> throwErrorListener = std::make_unique<ThrowErrorListener>();
    std::shared_ptr<CustomErrorStrategy> customErrorStrategy = std::make_shared<CustomErrorStrategy>();

    sqlLexer.removeErrorListeners();
    sqlLexer.addErrorListener(throwErrorListener.get());

    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
    parser.removeErrorListeners();
    parser.addErrorListener(throwErrorListener.get());
    parser.setErrorHandler(customErrorStrategy);
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

    bool usingWhere = false;
    bool usingGroupBy = false;
    bool usingOrderBy = false;
    bool usingAggregation = false;
    bool usingJoin = false;
    bool usingLoad = false;
    bool nonSelect = true;
    if (statement->sqlSelect())
    {
        WalkSqlSelect(walker, gpuSqlListener, cpuWhereListener, statement->sqlSelect(), usingWhere,
                      usingGroupBy, usingOrderBy, usingAggregation, usingJoin, usingLoad, nonSelect);
    }
    else if (statement->showStatement())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->showStatement());
    }
    else if (statement->showQueryTypes())
    {
        WalkSqlSelect(walker, gpuSqlListener, cpuWhereListener,
                      statement->showQueryTypes()->sqlSelect(), usingWhere, usingGroupBy,
                      usingOrderBy, usingAggregation, usingJoin, usingLoad, nonSelect);
        isSingleGpuStatement_ = true;
        gpuSqlListener.exitShowQueryTypes(statement->showQueryTypes());
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
    else if (statement->sqlAlterDatabase())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlAlterDatabase());
    }
    else if (statement->sqlCreateIndex())
    {
        isSingleGpuStatement_ = true;
        walker.walk(&gpuSqlListener, statement->sqlCreateIndex());
    }
    if (wasAborted_)
    {
        return nullptr;
    }
    int32_t threadCount = isSingleGpuStatement_ ? 1 : context.getDeviceCount();

    GpuSqlDispatcher::ResetGroupByCounters();
    GpuSqlDispatcher::ResetOrderByCounters();

    std::vector<std::thread> dispatcherFutures;
    std::vector<std::exception_ptr> dispatcherExceptions;
    std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults;
    std::vector<std::string> lockList;
    if (database_)
    {
        const std::string dbName = database_->GetName();
        for (auto& tableName : GpuSqlDispatcher::linkTable)
        {
            lockList.push_back(dbName + "." + tableName.first);
            lockList.push_back(dbName + "." + tableName.first + GpuSqlDispatcher::NULL_SUFFIX);
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
        return nullptr;
    }

    auto ret =
        (MergeDispatcherResults(dispatcherResults, gpuSqlListener.GetAliasList(), gpuSqlListener.ResultLimit,
                                gpuSqlListener.ResultOffset, usingWhere, usingGroupBy, usingOrderBy,
                                usingAggregation, usingJoin, usingLoad, nonSelect));

    for (auto& column : gpuSqlListener.ColumnOrder)
    {
        std::string colName = column.second.front() == '$' ? column.second.substr(1) : column.second;
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(ret.get())->add_columnorder(colName);
    }

    return ret;
}

void GpuSqlCustomParser::WalkSqlSelect(antlr4::tree::ParseTreeWalker& walker,
                                       GpuSqlListener& gpuSqlListener,
                                       CpuWhereListener& cpuWhereListener,
                                       GpuSqlParser::SqlSelectContext* sqlSelectContext,
                                       bool& usingWhere,
                                       bool& usingGroupBy,
                                       bool& usingOrderBy,
                                       bool& usingAggregation,
                                       bool& usingJoin,
                                       bool& usingLoad,
                                       bool& nonSelect)
{
    if (database_ == nullptr)
    {
        throw DatabaseNotUsedException();
    }

    int32_t columnOrder = 0;
    std::vector<std::pair<int32_t, GpuSqlParser::SelectColumnContext*>> aggColumns;
    std::vector<std::pair<int32_t, GpuSqlParser::SelectColumnContext*>> nonAggColumns;

    for (auto column : sqlSelectContext->selectColumns()->selectColumn())
    {
        if (ContainsAggregation(column))
        {
            aggColumns.push_back({columnOrder++, column});
        }
        else
        {
            nonAggColumns.push_back({columnOrder++, column});
        }

        if (ContainsColumnId(column))
        {
            usingLoad = true;
        }
    }

    if (sqlSelectContext->offset())
    {
        walker.walk(&gpuSqlListener, sqlSelectContext->offset());
    }

    if (sqlSelectContext->limit())
    {
        walker.walk(&gpuSqlListener, sqlSelectContext->limit());
    }

    usingWhere = sqlSelectContext->whereClause() != nullptr;
    usingGroupBy = sqlSelectContext->groupByColumns() != nullptr;
    usingOrderBy = sqlSelectContext->orderByColumns() != nullptr;
    usingAggregation = !aggColumns.empty();
    usingJoin = sqlSelectContext->joinClauses() != nullptr;
    usingLoad = usingLoad || (sqlSelectContext->selectColumns()->selectAllColumns().size() > 0);
    nonSelect = false;

    gpuSqlListener.SetContainsAggFunction(usingAggregation);

    gpuSqlListener.LimitOffset(usingWhere, usingGroupBy, usingOrderBy, usingAggregation, usingJoin, usingLoad);

    walker.walk(&gpuSqlListener, sqlSelectContext->fromTables());
    walker.walk(&cpuWhereListener, sqlSelectContext->fromTables());

    gpuSqlListener.ExtractColumnAliasContexts(sqlSelectContext->selectColumns());
    gpuSqlListener.LockAliasRegisters();
    cpuWhereListener.ExtractColumnAliasContexts(sqlSelectContext->selectColumns());

    if (sqlSelectContext->joinClauses())
    {
        walker.walk(&gpuSqlListener, sqlSelectContext->joinClauses());
        walker.walk(&cpuWhereListener, sqlSelectContext->joinClauses());
        joinDispatcher_->Execute();
    }

    if (sqlSelectContext->whereClause())
    {
        walker.walk(&gpuSqlListener, sqlSelectContext->whereClause());
        walker.walk(&cpuWhereListener, sqlSelectContext->whereClause());
    }

    if (sqlSelectContext->groupByColumns())
    {
        walker.walk(&gpuSqlListener, sqlSelectContext->groupByColumns());
    }

    for (auto column : sqlSelectContext->selectColumns()->selectAllColumns())
    {
        gpuSqlListener.exitSelectAllColumns(column);
        break;
    }

    for (auto column : aggColumns)
    {
        gpuSqlListener.CurrentSelectColumnIndex = column.first;
        walker.walk(&gpuSqlListener, column.second);
    }

    for (auto column : nonAggColumns)
    {
        gpuSqlListener.CurrentSelectColumnIndex = column.first;
        walker.walk(&gpuSqlListener, column.second);
    }

    if (sqlSelectContext->orderByColumns())
    {
        gpuSqlListener.enterOrderByColumns(sqlSelectContext->orderByColumns());

        // flip the order of ORDER BY columns
        std::vector<GpuSqlParser::OrderByColumnContext*> orderByColumns;

        for (auto orderByCol : sqlSelectContext->orderByColumns()->orderByColumn())
        {
            orderByColumns.insert(orderByColumns.begin(), orderByCol);
        }

        for (auto orderByCol : orderByColumns)
        {
            walker.walk(&gpuSqlListener, orderByCol);
        }

        gpuSqlListener.exitOrderByColumns(sqlSelectContext->orderByColumns());
    }

    gpuSqlListener.exitSelectColumns(sqlSelectContext->selectColumns());
}

void GpuSqlCustomParser::InterruptQueryExecution()
{
    BOOST_LOG_TRIVIAL(debug) << "GpuSqlCustomParser: Aborting parser has started...";
    for (auto& dispatcher : dispatchers_)
    {
        dispatcher->Abort();
    }
    if (joinDispatcher_)
    {
        joinDispatcher_->Abort();
    }
    wasAborted_ = true;
    BOOST_LOG_TRIVIAL(debug) << "GpuSqlCustomParser: Aborting parser has finnished successfully.";
}

/// Merges partial dispatcher respnse messages to final response message
/// <param="dispatcherResults">Partial dispatcher result messages</param>
/// <param="resultLimit">Row limit</param>
/// <param="resultOffset">Row offset</param>
/// <returns="reponseMessage">Merged response message</returns>
std::unique_ptr<google::protobuf::Message>
GpuSqlCustomParser::MergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults,
                                           const std::unordered_map<std::string, std::string>& aliasTable,
                                           int64_t resultLimit,
                                           int64_t resultOffset,
                                           bool usingWhere,
                                           bool usingGroupBy,
                                           bool usingOrderBy,
                                           bool usingAggregation,
                                           bool usingJoin,
                                           bool usingLoad,
                                           bool nonSelect)
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Limit: " << resultLimit << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Offset: " << resultOffset << '\n';

    std::unique_ptr<QikkDB::NetworkClient::Message::QueryResponseMessage> responseMessage =
        std::make_unique<QikkDB::NetworkClient::Message::QueryResponseMessage>();
    for (auto& partialResult : dispatcherResults)
    {
        QikkDB::NetworkClient::Message::QueryResponseMessage* partialMessage =
            dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(partialResult.get());
        for (auto& partialPayload : partialMessage->payloads())
        {
            std::string key = partialPayload.first;
            QikkDB::NetworkClient::Message::QueryResponsePayload payload = partialPayload.second;

            int64_t payloadSize;
            switch (payload.payload_case())
            {
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
                payloadSize = payload.intpayload().intdata_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
                payloadSize = payload.int64payload().int64data_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
                payloadSize = payload.floatpayload().floatdata_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
                payloadSize = payload.doublepayload().doubledata_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
                payloadSize = payload.stringpayload().stringdata_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
                payloadSize = payload.pointpayload().pointdata_size();
                break;
            case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
                payloadSize = payload.polygonpayload().polygondata_size();
                break;
            default:
                break;
            }

            if (partialMessage->nullbitmasks().find(key) != partialMessage->nullbitmasks().end())
            {
                std::vector<nullmask_t> partialBitMask(
                    partialMessage->nullbitmasks().at(key).nullmask().begin(),
                    partialMessage->nullbitmasks().at(key).nullmask().end());
                GpuSqlDispatcher::MergePayloadBitmask(key, responseMessage.get(), partialBitMask, payloadSize);
            }
            GpuSqlDispatcher::MergePayload(key, aliasTable.at(key), responseMessage.get(), payload);
        }
    }

    if (usingWhere || usingGroupBy || usingOrderBy || usingAggregation || usingJoin || !usingLoad || nonSelect)
    {
        TrimResponseMessage(responseMessage.get(), resultLimit, resultOffset);
    }
    return std::move(responseMessage);
}

/// Trims all payloads of result message according to limit and offset
/// <param="responseMessage">Response message to be trimmed</param>
/// <param="limit">Row limit</param>
/// <param="offset">Row offset</param>
void GpuSqlCustomParser::TrimResponseMessage(google::protobuf::Message* responseMessage, int64_t limit, int64_t offset)
{
    auto queryResponseMessage =
        dynamic_cast<QikkDB::NetworkClient::Message::QueryResponseMessage*>(responseMessage);
    for (auto& queryPayload : *queryResponseMessage->mutable_payloads())
    {
        std::string key = queryPayload.first;
        QikkDB::NetworkClient::Message::QueryResponsePayload& payload = queryPayload.second;
        int64_t payloadSize = 0;
        TrimPayload(payload, limit, offset, payloadSize);
        if (queryResponseMessage->nullbitmasks().find(key) != queryResponseMessage->nullbitmasks().end())
        {
            QikkDB::NetworkClient::Message::QueryNullmaskPayload& nullMaskPayload =
                queryResponseMessage->mutable_nullbitmasks()->at(key);
            TrimNullMaskPayload(nullMaskPayload, limit, offset, payloadSize);
        }
    }
}

/// Trims single payload of result message according to limit and offset
/// <param="payload">Payload to be trimmed</param>
/// <param="limit">Row limit</param>
/// <param="offset">Row offset</param>
void GpuSqlCustomParser::TrimPayload(QikkDB::NetworkClient::Message::QueryResponsePayload& payload,
                                     int64_t limit,
                                     int64_t offset,
                                     int64_t& payloadSize)
{
    switch (payload.payload_case())
    {
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
    {
        payloadSize = payload.intpayload().intdata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_intpayload()->mutable_intdata()->begin();
        payload.mutable_intpayload()->mutable_intdata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_intpayload()->mutable_intdata()->begin();
        auto end = payload.mutable_intpayload()->mutable_intdata()->end();
        payload.mutable_intpayload()->mutable_intdata()->erase(begin + clampedLimit, end);
    }
    break;

    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
    {
        payloadSize = payload.floatpayload().floatdata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_floatpayload()->mutable_floatdata()->begin();
        payload.mutable_floatpayload()->mutable_floatdata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_floatpayload()->mutable_floatdata()->begin();
        auto end = payload.mutable_floatpayload()->mutable_floatdata()->end();
        payload.mutable_floatpayload()->mutable_floatdata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
    {
        payloadSize = payload.int64payload().int64data().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_int64payload()->mutable_int64data()->begin();
        payload.mutable_int64payload()->mutable_int64data()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_int64payload()->mutable_int64data()->begin();
        auto end = payload.mutable_int64payload()->mutable_int64data()->end();
        payload.mutable_int64payload()->mutable_int64data()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDateTimePayload:
    {
        payloadSize = payload.datetimepayload().datetimedata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_datetimepayload()->mutable_datetimedata()->begin();
        payload.mutable_datetimepayload()->mutable_datetimedata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_datetimepayload()->mutable_datetimedata()->begin();
        auto end = payload.mutable_datetimepayload()->mutable_datetimedata()->end();
        payload.mutable_datetimepayload()->mutable_datetimedata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
    {
        payloadSize = payload.doublepayload().doubledata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_doublepayload()->mutable_doubledata()->begin();
        payload.mutable_doublepayload()->mutable_doubledata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_doublepayload()->mutable_doubledata()->begin();
        auto end = payload.mutable_doublepayload()->mutable_doubledata()->end();
        payload.mutable_doublepayload()->mutable_doubledata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
    {
        payloadSize = payload.pointpayload().pointdata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_pointpayload()->mutable_pointdata()->begin();
        payload.mutable_pointpayload()->mutable_pointdata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_pointpayload()->mutable_pointdata()->begin();
        auto end = payload.mutable_pointpayload()->mutable_pointdata()->end();
        payload.mutable_pointpayload()->mutable_pointdata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
    {
        payloadSize = payload.polygonpayload().polygondata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_polygonpayload()->mutable_polygondata()->begin();
        payload.mutable_polygonpayload()->mutable_polygondata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_polygonpayload()->mutable_polygondata()->begin();
        auto end = payload.mutable_polygonpayload()->mutable_polygondata()->end();
        payload.mutable_polygonpayload()->mutable_polygondata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
    {
        payloadSize = payload.stringpayload().stringdata().size();
        int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
        int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

        auto begin = payload.mutable_stringpayload()->mutable_stringdata()->begin();
        payload.mutable_stringpayload()->mutable_stringdata()->erase(begin, begin + clampedOffset);

        begin = payload.mutable_stringpayload()->mutable_stringdata()->begin();
        auto end = payload.mutable_stringpayload()->mutable_stringdata()->end();
        payload.mutable_stringpayload()->mutable_stringdata()->erase(begin + clampedLimit, end);
    }
    break;
    case QikkDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::PAYLOAD_NOT_SET:
        break;
    }
}

void GpuSqlCustomParser::TrimNullMaskPayload(QikkDB::NetworkClient::Message::QueryNullmaskPayload& payload,
                                             const int64_t limit,
                                             const int64_t offset,
                                             const int64_t payloadSize)
{
    const int64_t clampedOffset = std::clamp<int64_t>(offset, 0, payloadSize);
    const int64_t clampedLimit = std::clamp<int64_t>(limit, 0, payloadSize - clampedOffset);

    GpuSqlDispatcher::ShiftNullMaskLeft(*payload.mutable_nullmask(), clampedOffset);

    payload.mutable_nullmask()->erase(payload.mutable_nullmask()->begin() +
                                          NullValues::GetNullBitMaskSize(clampedLimit),
                                      payload.mutable_nullmask()->end());
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

bool GpuSqlCustomParser::ContainsColumnId(GpuSqlParser::SelectColumnContext* ctx)
{
    antlr4::tree::ParseTreeWalker walker;

    class : public GpuSqlParserBaseListener
    {
    public:
        bool containsColumnId = false;

    private:
        void exitVarReference(GpuSqlParser::VarReferenceContext* ctx) override
        {
            containsColumnId = true;
        }

    } findColListener;

    walker.walk(&findColListener, ctx);

    return findColListener.containsColumnId;
}

void ThrowErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                     antlr4::Token* offendingSymbol,
                                     size_t line,
                                     size_t charPositionInLine,
                                     const std::string& msg,
                                     std::exception_ptr e)
{
    const std::string finalMsg =
        "Error: line " + std::to_string(line) + ":" + std::to_string(charPositionInLine) + " " + msg;

    BOOST_LOG_TRIVIAL(debug) << finalMsg;
    throw antlr4::ParseCancellationException(finalMsg);
}

void CustomErrorStrategy::reportInputMismatch(antlr4::Parser* recognizer, const antlr4::InputMismatchException& e)
{
    const std::string badSymbol =
        e.getOffendingToken() == nullptr ? std::string("") : e.getOffendingToken()->getText();

    std::string msg = "mismatched input " + getTokenErrorDisplay(e.getOffendingToken()) +
                      " expecting " + e.getExpectedTokens().toString(recognizer->getVocabulary()) +
                      " near symbol '" + badSymbol + "'.";

    // add more custom semantic check error messages here

    if (dynamic_cast<GpuSqlParser::UnaryOperationContext*>(e.getCtx()) &&
        dynamic_cast<GpuSqlParser::UnaryOperationContext*>(e.getCtx())->LPAREN() == nullptr &&
        dynamic_cast<GpuSqlParser::UnaryOperationContext*>(e.getCtx())->expression() == nullptr &&
        dynamic_cast<GpuSqlParser::UnaryOperationContext*>(e.getCtx())->RPAREN() == nullptr)
    {
        auto ctx = dynamic_cast<GpuSqlParser::UnaryOperationContext*>(e.getCtx());
        msg += " Incomplete call of unary function: '" + ctx->op->getText() +
               "' or ambiguous column name detected (confused with "
               "function name). Complete the function call with parenthesis and arguments or use "
               "delimited identifier for a column.";
    }

    else if (dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx()) &&
             dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx())->LPAREN() == nullptr &&
             dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx())->left == nullptr &&
             dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx())->right == nullptr &&
             dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx())->RPAREN() == nullptr)
    {
        auto ctx = dynamic_cast<GpuSqlParser::BinaryOperationContext*>(e.getCtx());
        msg += " Incomplete call of binary function: '" + ctx->op->getText() +
               "' or ambiguous column name detected (confused with "
               "function name). Complete the function call with parenthesis and arguments or use "
               "delimited identifier for a column.";
    }

    else if (dynamic_cast<GpuSqlParser::AggregationContext*>(e.getCtx()) &&
             dynamic_cast<GpuSqlParser::AggregationContext*>(e.getCtx())->LPAREN() == nullptr &&
             dynamic_cast<GpuSqlParser::AggregationContext*>(e.getCtx())->expression() == nullptr &&
             dynamic_cast<GpuSqlParser::AggregationContext*>(e.getCtx())->RPAREN() == nullptr)
    {
        auto ctx = dynamic_cast<GpuSqlParser::AggregationContext*>(e.getCtx());
        msg += " Incomplete call of aggregation function: '" + ctx->op->getText() +
               "' or ambiguous column name detected (confused with "
               "function name). Complete the function call with parenthesis and arguments or use "
               "delimited identifier for a column.";
    }
    recognizer->notifyErrorListeners(e.getOffendingToken(), msg, std::make_exception_ptr(e));
}
