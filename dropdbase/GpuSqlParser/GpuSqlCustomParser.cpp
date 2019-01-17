//
// Created by Martin Staňo on 2019-01-14.
//

#include "GpuSqlCustomParser.h"

//TODO:parse()

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query) : database(
        database), query(query)
{}


void GpuSqlCustomParser::parse()
{
    antlr4::ANTLRInputStream sqlInputStream(query);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
    GpuSqlParser::SqlFileContext *sqlFileContext = parser.sqlFile();

    antlr4::tree::ParseTreeWalker walker;


    for (auto statement : sqlFileContext->statement())
    {
        GpuSqlDispatcher dispatcher;
        GpuSqlListener gpuSqlListener(database, dispatcher);
        if (statement->sqlSelect())
        {
            if (database == nullptr)
            {
                throw DatabaseNotFoundException();
            }

            walker.walk(&gpuSqlListener, statement->sqlSelect()->fromTables());

            if (statement->sqlSelect()->whereClause())
            {
                walker.walk(&gpuSqlListener, statement->sqlSelect()->whereClause());
            }

            if(statement->sqlSelect()->groupByColumns())
            {
                walker.walk(&gpuSqlListener, statement->sqlSelect()->groupByColumns());
            }

            walker.walk(&gpuSqlListener, statement->sqlSelect()->selectColumns());

            if(statement->sqlSelect()->offset())
            {
                walker.walk(&gpuSqlListener, statement->sqlSelect()->offset());
            }

            if(statement->sqlSelect()->limit())
            {
                walker.walk(&gpuSqlListener, statement->sqlSelect()->limit());
            }

            if(statement->sqlSelect()->orderByColumns())
            {
                walker.walk(&gpuSqlListener, statement->sqlSelect()->orderByColumns());
            }
        }
        dispatcher.execute();
    }
}
