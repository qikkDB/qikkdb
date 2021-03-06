//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#include "GpuSqlParser.h"

#include <string>
#include <memory>

#include "GpuSqlDispatcher.h"
#include "GpuSqlJoinDispatcher.h"


class Database;

class GpuSqlJoinDispatcher;

class GpuSqlListener;

class CpuWhereListener;

namespace QikkDB
{
namespace NetworkClient
{
namespace Message
{
class QueryResponsePayload;
}
} // namespace NetworkClient
} // namespace QikkDB

namespace google
{
namespace protobuf
{
class Message;
}
} // namespace google

class GpuSqlCustomParser
{

private:
    const std::shared_ptr<Database>& database_;
    void TrimResponseMessage(google::protobuf::Message* responseMessage, int64_t limit, int64_t offset);
    void TrimPayload(QikkDB::NetworkClient::Message::QueryResponsePayload& payload, int64_t limit, int64_t offset, int64_t& payloadSize);
    void TrimNullMaskPayload(QikkDB::NetworkClient::Message::QueryNullmaskPayload& payload,
                             const int64_t limit,
                             const int64_t offset,
                             const int64_t payloadSize);
    bool isSingleGpuStatement_;
    bool wasAborted_;
    std::string query_;
    std::unique_ptr<google::protobuf::Message>
    MergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults,
                           const std::unordered_map<std::string, std::string>& aliasTable,
                           int64_t resultLimit,
                           int64_t resultOffset,
                           bool usingWhere,
                           bool usingGroupBy,
                           bool usingOrderBy,
                           bool usingAggregation,
                           bool usingJoin,
                           bool usingLoad,
                           bool nonSelect);

    std::vector<std::unique_ptr<GpuSqlDispatcher>> dispatchers_;
    std::unique_ptr<GpuSqlJoinDispatcher> joinDispatcher_;
    void WalkSqlSelect(antlr4::tree::ParseTreeWalker& walker,
                       GpuSqlListener& gpuSqlListener,
                       CpuWhereListener& cpuWhereListener,
                       GpuSqlParser::SqlSelectContext* sqlSelectContext,
                       bool& usingWhere,
                       bool& usingGroupBy,
                       bool& usingOrderBy,
                       bool& usingAggregation,
                       bool& usingJoin,
                       bool& usingLoad,
                       bool& nonSelect);

public:
    GpuSqlCustomParser(const std::shared_ptr<Database>& database, const std::string& query);

    std::unique_ptr<google::protobuf::Message> Parse();
    void InterruptQueryExecution();
    bool ContainsAggregation(GpuSqlParser::SelectColumnContext* ctx);
    bool ContainsColumnId(GpuSqlParser::SelectColumnContext* ctx);
};

class ThrowErrorListener : public antlr4::BaseErrorListener
{
public:
    void syntaxError(antlr4::Recognizer* recognizer,
                     antlr4::Token* offendingSymbol,
                     size_t line,
                     size_t charPositionInLine,
                     const std::string& msg,
                     std::exception_ptr e) override;
};

class CustomErrorStrategy : public antlr4::DefaultErrorStrategy
{
public:
    void reportInputMismatch(antlr4::Parser* recognizer, const antlr4::InputMismatchException& e) override;
};


#endif // DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
