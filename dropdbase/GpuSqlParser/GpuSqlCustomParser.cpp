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
#include "../QueryEngine/Context.h"
#include <iostream>
#include <future>
#include <thread>

GpuSqlCustomParser::GpuSqlCustomParser(const std::shared_ptr<Database> &database, const std::string &query) : 
	database(database),
	query(query),
	isSingleGpuStatement(false)
{}


std::unique_ptr<google::protobuf::Message> GpuSqlCustomParser::parse()
{
	Context& context = Context::getInstance();

	antlr4::ANTLRInputStream sqlInputStream(query);
	GpuSqlLexer sqlLexer(&sqlInputStream);
	antlr4::CommonTokenStream commonTokenStream(&sqlLexer);
	GpuSqlParser parser(&commonTokenStream);
	parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
	parser.getInterpreter<antlr4::atn::ParserATNSimulator>()->setPredictionMode(antlr4::atn::PredictionMode::SLL);
	GpuSqlParser::StatementContext *statement = parser.statement();

	antlr4::tree::ParseTreeWalker walker;

	std::vector<std::unique_ptr<IGroupBy>> groupByInstances;

	for (int i = 0; i < context.getDeviceCount(); i++)
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

		if (statement->sqlSelect()->groupByColumns())
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

		if (statement->sqlSelect()->offset())
		{
			walker.walk(&gpuSqlListener, statement->sqlSelect()->offset());
		}

		if (statement->sqlSelect()->limit())
		{
			walker.walk(&gpuSqlListener, statement->sqlSelect()->limit());
		}

		if (statement->sqlSelect()->orderByColumns())
		{
			walker.walk(&gpuSqlListener, statement->sqlSelect()->orderByColumns());
		}
	}
	else if (statement->showStatement())
	{
		isSingleGpuStatement = true;
		walker.walk(&gpuSqlListener, statement->showStatement());
	}
	else if (statement->sqlInsertInto())
	{
		isSingleGpuStatement = true;
		walker.walk(&gpuSqlListener, statement->sqlInsertInto());
	}

	int32_t threadCount = isSingleGpuStatement ? 1 : context.getDeviceCount();

	GpuSqlDispatcher::ResetGroupByCounters();
	std::vector<std::unique_ptr<GpuSqlDispatcher>> dispatchers;
	std::vector<std::thread> dispatcherFutures;
	std::vector<std::exception_ptr> dispatcherExceptions;
	std::vector<std::unique_ptr<google::protobuf::Message>> dispatcherResults; 

	for (int i = 0; i < threadCount; i++)
	{
		dispatcherResults.emplace_back(nullptr);
		dispatcherExceptions.emplace_back(nullptr);
	}

	for (int i = 0; i < threadCount; i++)
	{
		dispatchers.emplace_back(std::make_unique<GpuSqlDispatcher>(database, groupByInstances, i));
		dispatcher.get()->copyExecutionDataTo(*dispatchers[i]);
		dispatcherFutures.push_back(std::thread(std::bind(&GpuSqlDispatcher::execute, dispatchers[i].get(), std::ref(dispatcherResults[i]), std::ref(dispatcherExceptions[i]))));
	}

	for (int i = 0; i < threadCount; i++)
	{
		dispatcherFutures[i].join();
		std::cout << "TID: " << i << " Done \n";
	}

	for (int i = 0; i < threadCount; i++)
	{
		if (dispatcherExceptions[i])
		{
			std::rethrow_exception(dispatcherExceptions[i]);
		}
	}
	
		
	return std::move(mergeDispatcherResults(dispatcherResults, gpuSqlListener.resultLimit, gpuSqlListener.resultOffset));
}


std::unique_ptr<google::protobuf::Message>
GpuSqlCustomParser::mergeDispatcherResults(std::vector<std::unique_ptr<google::protobuf::Message>>& dispatcherResults,int32_t resultLimit, int32_t resultOffset)
{
    std::cout << "Limit: " << resultLimit << std::endl;
    std::cout << "Offset: " << resultOffset << std::endl;

	std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> responseMessage = std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>();
	for (auto& partialResult : dispatcherResults)
	{
		ColmnarDB::NetworkClient::Message::QueryResponseMessage* partialMessage = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(partialResult.get());
		for (auto& partialPayload : partialMessage->payloads())
		{
			std::string key = partialPayload.first;
			ColmnarDB::NetworkClient::Message::QueryResponsePayload payload = partialPayload.second;
			GpuSqlDispatcher::MergePayload(key, responseMessage.get(), payload);
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
