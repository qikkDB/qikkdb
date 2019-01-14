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
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/initializer.h>

class GpuSqlCustomParser {

private:
    Database database;
    log4cplus::Initializer initializer;
    log4cplus::BasicConfigurator config;

public:
    GpuSqlCustomParser(const Database &database, const std::string &query);

};


#endif //DROPDBASE_INSTAREA_GPUSQLCUSTOMPARSER_H
