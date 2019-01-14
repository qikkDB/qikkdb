//
// Created by Martin Staňo on 2019-01-14.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
#define DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H

#pragma once
#include "GpuSqlParser/GpuSqlParser.h"
#include "GpuSqlParser/GpuSqlLexer.h"
#include "Database.h"
#include <string>

class GpuSqlCustomParser {

private:
    Database database;

public:
    GpuSqlCustomParser(const Database &database, const std::string &query);

};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
