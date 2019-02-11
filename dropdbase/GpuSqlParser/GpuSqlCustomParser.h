//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include <google/protobuf/message.h>
#include "../Database.h"
#include "GpuSqlParser.h"
#include <string>
#include <memory>

class GpuSqlCustomParser
{

private:
    const std::shared_ptr<Database> &database;
    std::string query;
	std::unique_ptr<google::protobuf::Message> mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults);

public:
    GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query);

	std::unique_ptr<google::protobuf::Message> parse();
	bool containsAggregation(GpuSqlParser::SelectColumnContext *ctx);
};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
