//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include <google/protobuf/message.h>
#include "../messages/QueryResponseMessage.pb.h"
#include "../Database.h"
#include "GpuSqlParser.h"
#include <string>
#include <memory>

class GpuSqlCustomParser
{

private:
    const std::shared_ptr<Database> &database;
	void trimResponseMessage(google::protobuf::Message* responseMessage, int32_t limit, int32_t offset);
	void trimPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload, int32_t limit, int32_t offset);
	std::pair<int32_t, int32_t> getClampedLimitOffset(int32_t payloadSize, int32_t limit, int32_t offset);
	bool isSingleGpuStatement;
    std::string query;
	std::unique_ptr<google::protobuf::Message> mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults, int32_t resultLimit, int32_t resultOffset);

public:
    GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query);

	std::unique_ptr<google::protobuf::Message> parse();
	bool containsAggregation(GpuSqlParser::SelectColumnContext *ctx);
};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
