#pragma once
#include <thread>
#include <mutex>
#include <google/protobuf/message.h>
#include <future>
#include "IClientHandler.h"
#include "Database.h"
#include <memory>

class TCPClientHandler final : public IClientHandler
{
private:
	const int FRAGMENT_SIZE = 1000;
	int sentRecords_;
	int lastResultLen_;
	std::mutex queryMutex_;
	std::unique_ptr<google::protobuf::Message> GetNextQueryResult();
	std::future<std::unique_ptr<google::protobuf::Message>> lastQueryResult_;
	std::unique_ptr<google::protobuf::Message> lastResultMessage_;
	std::unique_ptr<google::protobuf::Message> RunQuery(const std::weak_ptr<Database>& database, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage);

public:
	TCPClientHandler() {};

	// Inherited via IClientHandler
	virtual std::unique_ptr<google::protobuf::Message> HandleInfoMessage(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::InfoMessage & infoMessage) override;
	virtual std::unique_ptr<google::protobuf::Message> HandleQuery(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage) override;
	virtual std::unique_ptr<google::protobuf::Message> HandleCSVImport(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::CSVImportMessage & csvImportMessage) override;
	virtual std::unique_ptr<google::protobuf::Message> HandleSetDatabase(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::SetDatabaseMessage & setDatabaseMessage) override;
};

