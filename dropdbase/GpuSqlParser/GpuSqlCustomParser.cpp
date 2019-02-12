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
#include "../QueryEngine/GPUCore/IGroupBy.h"
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
	const int32_t numOfDevices = 1;

	std::vector<std::unique_ptr<IGroupBy>> groupByInstances;

	for (int i = 0; i < numOfDevices; i++)
	{
		groupByInstances.emplace_back(nullptr);
	}

	std::unique_ptr<GpuSqlDispatcher> dispatcher = std::make_unique<GpuSqlDispatcher>(database, groupByInstances, -1);
	GpuSqlListener gpuSqlListener(database, *dispatcher);

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

	std::vector<std::unique_ptr<GpuSqlDispatcher>> dispatchers;
	std::vector<std::future<std::unique_ptr<google::protobuf::Message>>> dispatcherFutures;
	std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults;

	for (int i = 0; i < numOfDevices; i++)
	{
		dispatchers.emplace_back(std::make_unique<GpuSqlDispatcher>(database, groupByInstances, i));
		dispatcher.get()->copyExecutionDataTo(*dispatchers[i]);
		dispatcherFutures.push_back(std::async(std::launch::async, &GpuSqlDispatcher::execute, dispatchers[i].get()));
	}

	for (int i = 0; i < numOfDevices; i++)
	{
		dispatcherResults.push_back(std::move(dispatcherFutures[i].get()));
	}
		
    return std::move(mergeDispatcherResults(dispatcherResults));
}


std::unique_ptr<google::protobuf::Message> GpuSqlCustomParser::mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults)
{
	std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> responseMessage = std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>();
	for (auto& partialResult : dispatcherResults)
	{
		ColmnarDB::NetworkClient::Message::QueryResponseMessage* partialMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(partialResult.get());
		for (auto& partialPayload : partialMessage->payloads())
		{
			std::string key = partialPayload.first;
			ColmnarDB::NetworkClient::Message::QueryResponsePayload payload = partialPayload.second;
			if (responseMessage->payloads().find(key) == responseMessage->payloads().end())
			{
				responseMessage->mutable_payloads()->insert({ key, payload });
			}
			else
			{
				responseMessage->mutable_payloads()->at(key).MergeFrom(payload);
			}
		}
	}
	return std::move(responseMessage);
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
