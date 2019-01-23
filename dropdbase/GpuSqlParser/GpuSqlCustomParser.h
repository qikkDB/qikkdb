//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#include "GpuSqlParser.h"
#include "GpuSqlLexer.h"
#include "GpuSqlListener.h"
#include "ParserExceptions.h"
#include "QueryType.h"
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

    void parse();

};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
