#include "TCPClientHandler.h"
#include "ITCPWorker.h"
#include "CSVInMemoryImporter.h"
#include "Configuration.h"
#include <functional>
#include <stdexcept>

std::unique_ptr<google::protobuf::Message> TCPClientHandler::GetNextQueryResult()
{
	return std::unique_ptr<google::protobuf::Message>();
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::RunQuery(Database & database, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage)
{
	return std::unique_ptr<google::protobuf::Message>();
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleInfoMessage(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::InfoMessage & infoMessage)
{
	if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END)
	{
		worker.Abort();
	}
	else if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
	{
		return GetNextQueryResult();
	}
	else
	{
		//Log.WarnFormat("Invalid InfoMessage received, Code = {0}", infoMessage.Code);
	}
	return nullptr;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleQuery(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage)
{
	sentRecords_ = 0;
	lastResultLen_ = 0;
	lastQueryResult_ = std::async(std::launch::async, std::bind(&TCPClientHandler::RunQuery, this, *worker.currentDatabase_, queryMessage));
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);
	resultMessage->set_message("");
	return resultMessage;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleCSVImport(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::CSVImportMessage & csvImportMessage)
{
	CSVInMemoryImporter dataImporter(csvImportMessage.csvname(), csvImportMessage.payload());
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	try
	{
		auto importDB = Database::GetDatabaseByName(csvImportMessage.databasename());
		if (importDB == nullptr)
		{
			importDB = std::make_shared<Database>(csvImportMessage.databasename(), Configuration::BlockSize());
			dataImporter.ImportTables(importDB);
			Database::AddToInMemoryDatabaseList(importDB);
		}
		else
		{
			dataImporter.ImportTables(importDB);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "CSVImport: " << e.what() << std::endl;
	}
	return resultMessage;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleSetDatabase(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::SetDatabaseMessage & setDatabaseMessage)
{
	worker.currentDatabase_ = Database::GetDatabaseByName(setDatabaseMessage.databasename());
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
	resultMessage->set_message("");
	if (worker.currentDatabase_ == nullptr)
	{
		resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
		resultMessage->set_message("No such database");
	}
	return resultMessage;
}
