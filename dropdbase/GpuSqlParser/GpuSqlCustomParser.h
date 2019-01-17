//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include "GpuSqlParser/GpuSqlParser.h"
#include "GpuSqlParser/GpuSqlLexer.h"
#include "Database.h"
#include "ParserExceptions.h"
#include "QueryType.h"
#include "GpuSqlListener.h"
#include <string>
#include <memory>

class GpuSqlCustomParser
{

private:
    std::shared_ptr<Database> database;
    std::string query;

public:
    GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query);

    void parse();

};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
