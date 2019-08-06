//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include "GpuSqlParser.h"
#include <string>
#include <memory>

class Database;

namespace ColmnarDB
{
namespace NetworkClient
{
namespace Message
{
class QueryResponsePayload;
}
} // namespace NetworkClient
} // namespace ColmnarDB

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
    void TrimPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload, int64_t limit, int64_t offset);
    bool isSingleGpuStatement_;
    std::string query_;
    std::unique_ptr<google::protobuf::Message>
    MergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults,
                           int64_t resultLimit,
                           int64_t resultOffset);

public:
    GpuSqlCustomParser(const std::shared_ptr<Database>& database, const std::string& query);

    std::unique_ptr<google::protobuf::Message> Parse();
    bool ContainsAggregation(GpuSqlParser::SelectColumnContext* ctx);
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


#endif // DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
