//
// Created by Martin Staňo on 2019-01-14.
//

#include "GpuSqlCustomParser.h"
#include "GpuSqlParser.h"
#include "GpuSqlLexer.h"
#include "GpuSqlListener.h"
#include "GpuSqlDispatcher.h"
#include "ParserExceptions.h"
#include "QueryType.h"
#include <iostream>
#include <future>
#include <thread>

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query) : 
	database(database),
	query(query)
{}


std::unique_ptr<google::protobuf::Message> GpuSqlCustomParser::parse()
{
    antlr4::ANTLRInputStream sqlInputStream(query);
    GpuSqlLexer sqlLexer(&sqlInputStream);
    antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
    GpuSqlParser parser(&commonTokenStream);
    parser.getInterpreter<antlr4::atn::ParserATNSimulator>()->setPredictionMode(antlr4::atn::PredictionMode::SLL);
    GpuSqlParser::StatementContext *statement = parser.statement();

    antlr4::tree::ParseTreeWalker walker;

	GpuSqlDispatcher dispatcher(database);
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

		std::vector<GpuSqlParser::SelectColumnContext*> aggColumns;
		std::vector<GpuSqlParser::SelectColumnContext*> nonAggColumns;


		for (auto column : statement->sqlSelect()->selectColumns()->selectColumn()) 
		{
			if (containsAggregation(column))
			{
				aggColumns.push_back(column);
			}
			else
			{
				nonAggColumns.push_back(column);
			}
		}

		for (auto column : aggColumns) 
		{
			walker.walk(&gpuSqlListener, column);
		}

		for (auto column : nonAggColumns)
		{
			walker.walk(&gpuSqlListener, column);
		}

		gpuSqlListener.exitSelectColumns(statement->sqlSelect()->selectColumns());

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

	std::vector<GpuSqlDispatcher> dispatchers;
	std::vector<std::future<std::unique_ptr<google::protobuf::Message>>> dispatcherFutures;
	std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults;

	const int32_t numOfDevices = 1;
	for (int i = 0; i < numOfDevices; i++)
	{
		dispatchers.push_back(dispatcher);
	}

	for (int i = 0; i < numOfDevices; i++)
	{
		dispatcherFutures.push_back(std::async(std::launch::async, &GpuSqlDispatcher::execute, &dispatchers[i]));
	}

	for (int i = 0; i < numOfDevices; i++)
	{
		dispatcherResults.push_back(dispatcherFutures[i].get());
	}
	
	mergeDispatcherResults(dispatcherResults);
    return std::move(dispatcherResults[0]);
}


std::unique_ptr<google::protobuf::Message> GpuSqlCustomParser::mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults)
{
	//TODO: merge
	return dispatcherResults[0];
}

bool GpuSqlCustomParser::containsAggregation(GpuSqlParser::SelectColumnContext * ctx)
{
	antlr4::tree::ParseTreeWalker walker;

	class : public GpuSqlParserBaseListener {
	public:
		bool containsAggregation = false;
	private:
		void exitAggregation(GpuSqlParser::AggregationContext *ctx) override 
		{
			containsAggregation = true;
		}

	} findAggListener;

	walker.walk(&findAggListener, ctx);

	return findAggListener.containsAggregation;
}
