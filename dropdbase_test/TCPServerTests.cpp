#include "gtest/gtest.h"
#include "../dropdbase/IClientHandler.h"
#include "../dropdbase/ClientPoolWorker.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/TCPServer.h"
#include <thread>

class DummyClientHandler : IClientHandler
{
	// Inherited via IClientHandler
	virtual std::unique_ptr<google::protobuf::Message> HandleInfoMessage(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::InfoMessage & infoMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> ret;
		if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
		{
			ColmnarDB::NetworkClient::Message::QueryResponsePayload qrp;
			(*qrp.mutable_stringpayload()->add_stringdata()) = "test";
			ret->mutable_payloads()->insert({"test", qrp});
			ret->mutable_timing()->insert({ "aaa",2 });
			return ret;
		}
		else
		{
			worker.Abort();
			return nullptr;
		}
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleQuery(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret;
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);
		ret->set_message("");
		return ret;
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleCSVImport(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::CSVImportMessage & csvImportMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret;
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
		ret->set_message("");
		return ret;
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleSetDatabase(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::SetDatabaseMessage & SetDatabaseMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret;
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
		ret->set_message("");
		return ret;
	}
};

TEST(TCPServer, ServerMessageInfo)
{
	TCPServer<DummyClientHandler,ClientPoolWorker> testServer("127.0.0.1", 12345);
	auto future = std::async(std::launch::async, [&testServer]() {testServer.Run(); });
	//ClientShit
	testServer.Abort();
	ASSERT_NO_THROW(future.get());
}