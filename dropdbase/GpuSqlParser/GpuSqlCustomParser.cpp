//
// Created by Martin Staňo on 2019-01-14.
//

#include "GpuSqlCustomParser.h"

GpuSqlCustomParser::GpuSqlCustomParser(const Database &database, const std::string &query) : database(database) {
    this->database = database;
    antlr4::ANTLRInputStream sqlInputStream(query);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
}
