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
	}
}

namespace google
{
	namespace protobuf
	{
		class Message;
	}
}

class GpuSqlCustomParser
{

private:
    const std::shared_ptr<Database> &database;
	void trimResponseMessage(google::protobuf::Message* responseMessage, int64_t limit, int64_t offset);
	void trimPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload, int64_t limit, int64_t offset);
	bool isSingleGpuStatement;
    std::string query;
	std::unique_ptr<google::protobuf::Message> mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults, int64_t resultLimit, int64_t resultOffset);

public:
    GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query);

	std::unique_ptr<google::protobuf::Message> parse();
	bool containsAggregation(GpuSqlParser::SelectColumnContext *ctx);
};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
