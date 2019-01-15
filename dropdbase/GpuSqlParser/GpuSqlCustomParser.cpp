//
// Created by Martin Staňo on 2019-01-14.
//

#include "GpuSqlCustomParser.h"

//TODO:parse()

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query) : database(
        database), query(query) {}


void GpuSqlCustomParser::parse() {
    antlr4::ANTLRInputStream sqlInputStream(query);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
    GpuSqlParser::SqlFileContext *sqlFileContext = parser.sqlFile();

    antlr4::tree::ParseTreeWalker walker;

    for (auto child : sqlFileContext->statement()) {
        if (child->sqlSelect()) {
            if (database == nullptr) {
                throw DatabaseNotFoundException();
            }
        }
    }
}
