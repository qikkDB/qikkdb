#include "gtest/gtest.h"
#include "../dropdbase/IClientHandler.h"
#include "../dropdbase/ClientPoolWorker.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/TCPServer.h"
#include "../dropdbase/NetworkMessage.h"
#include <boost/asio.hpp>
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

boost::asio::ip::tcp::socket connectToTestServer(boost::asio::io_context& context)
{
	boost::asio::ip::tcp::socket sock(context);
	boost::asio::connect(sock, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345));
}

TEST(TCPServer, ServerMessageInfo)
{
	TCPServer<DummyClientHandler,ClientPoolWorker> testServer("127.0.0.1", 12345);
	auto future = std::async(std::launch::async, [&testServer]() {testServer.Run(); });
	boost::asio::io_context context;
	auto sock = connectToTestServer(context);
	ColmnarDB::NetworkClient::Message::InfoMessage hello;
	hello.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_ESTABLISH);
	hello.set_message("");
	NetworkMessage::WriteToNetwork(hello, sock);
	auto ret = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(ret.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
	infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END);
	infoMessage.set_message("");
	NetworkMessage::WriteToNetwork(infoMessage, sock);
	testServer.Abort();
	ASSERT_NO_THROW(future.get());
}