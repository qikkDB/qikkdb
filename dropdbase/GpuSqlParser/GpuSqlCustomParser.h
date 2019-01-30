//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include <google/protobuf/message.h>
#include "../Database.h"
#include <string>
#include <memory>

class GpuSqlCustomParser
{

private:
    const std::shared_ptr<Database> &database;
    std::string query;

public:
    GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query);

	std::unique_ptr<google::protobuf::Message> parse();

};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
