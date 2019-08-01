#include "gtest/gtest.h"
#include "../dropdbase/IClientHandler.h"
#include "../dropdbase/ClientPoolWorker.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/messages/BulkImportMessage.pb.h"
#include "../dropdbase/messages/SetDatabaseMessage.pb.h"
#include "../dropdbase/TCPServer.h"
#include "../dropdbase/NetworkMessage.h"
#include <boost/asio.hpp>
#include <thread>
#include <stdexcept>

class DummyClientHandler : public IClientHandler
{
	// Inherited via IClientHandler
	virtual std::unique_ptr<google::protobuf::Message> HandleInfoMessage(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::InfoMessage & infoMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> ret = std::make_unique< ColmnarDB::NetworkClient::Message::QueryResponseMessage>();
		if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
		{
			ColmnarDB::NetworkClient::Message::QueryResponsePayload qrp;
			(*qrp.mutable_stringpayload()->add_stringdata()) = "test";
			ret->mutable_payloads()->insert({"test", qrp});
			ret->mutable_timing()->insert({ "aaa", 2 });
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
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret = std::make_unique< ColmnarDB::NetworkClient::Message::InfoMessage>();
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);
		ret->set_message("");
		return ret;
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleCSVImport(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::CSVImportMessage & csvImportMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret = std::make_unique< ColmnarDB::NetworkClient::Message::InfoMessage>();
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
		ret->set_message("");
		return ret;
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleSetDatabase(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::SetDatabaseMessage & SetDatabaseMessage) override
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret = std::make_unique< ColmnarDB::NetworkClient::Message::InfoMessage>();
		ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
		ret->set_message("");
		return ret;
	}
	virtual std::unique_ptr<google::protobuf::Message> HandleBulkImport(ITCPWorker& worker, const ColmnarDB::NetworkClient::Message::BulkImportMessage& bulkImportMessage, const char* dataBuffer, const char* nullMask) override 
	{
		std::unique_ptr<ColmnarDB::NetworkClient::Message::InfoMessage> ret = std::make_unique< ColmnarDB::NetworkClient::Message::InfoMessage>();
		if(std::string(bulkImportMessage.columnname()) != "test" || std::string(bulkImportMessage.tablename()) != "test" 
		|| bulkImportMessage.columntype() != ColmnarDB::NetworkClient::Message::DataType::COLUMN_INT || bulkImportMessage.elemcount() != 5)
		{
			printf("Something wrong.\n");
			ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);	
		}
		else
		{
			ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
			const int32_t* intData = reinterpret_cast<const int32_t*>(dataBuffer);
			for(int i = 0; i < 5; i++)
			{
				if(intData[i] != i+1)
				{
					printf("Data mismatch %d != %d.\n",intData[i],i+1);
					ret->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);	
					break;
				}
			}
			
		}
				
		ret->set_message("");
		return ret;
	}
};

boost::asio::ip::tcp::socket connectSocketToTestServer(boost::asio::io_context& context)
{
	boost::asio::ip::tcp::socket sock(context);
	boost::asio::ip::tcp::resolver resolver(context);
	auto endpoints = resolver.resolve("127.0.0.1", "12345");
	boost::asio::connect(sock, endpoints);
	return sock;
}

void connect(boost::asio::ip::tcp::socket& sock)
{
	ColmnarDB::NetworkClient::Message::InfoMessage hello;
	hello.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_ESTABLISH);
	hello.set_message("");
	NetworkMessage::WriteToNetwork(hello, sock);
	auto ret = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(ret.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
}

void disconnect(boost::asio::ip::tcp::socket& sock)
{
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END);
	infoMessage.set_message("");
	NetworkMessage::WriteToNetwork(infoMessage, sock);
}

void query(boost::asio::ip::tcp::socket& sock, const char* queryString)
{
	ColmnarDB::NetworkClient::Message::QueryMessage query;
	query.set_query(queryString);
	NetworkMessage::WriteToNetwork(query, sock);
	auto response = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(response.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);

}

ColmnarDB::NetworkClient::Message::QueryResponseMessage getNextQueryResult(boost::asio::ip::tcp::socket& sock)
{
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage_StatusCode_GET_NEXT_RESULT);
	infoMessage.set_message("");
	NetworkMessage::WriteToNetwork(infoMessage, sock);
	auto response = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::QueryResponseMessage ret;
	if (!response.UnpackTo(&ret))
	{
		throw std::domain_error("Invalid message received");
	}
	return ret;
}

void importCSV(boost::asio::ip::tcp::socket& sock, const char* name, const char* data)
{
	ColmnarDB::NetworkClient::Message::CSVImportMessage csvImport;
	csvImport.set_databasename(name);
	csvImport.set_payload(data);
	csvImport.set_csvname(name);
	NetworkMessage::WriteToNetwork(csvImport, sock);
	auto ret = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(ret.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
}

void setDatabase(boost::asio::ip::tcp::socket& sock, const char* name)
{
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDbMsg;
	setDbMsg.set_databasename(name);
	NetworkMessage::WriteToNetwork(setDbMsg, sock);
	auto response = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(response.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
}

void bulkImport(boost::asio::ip::tcp::socket& sock)
{
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage.set_elemcount(5);
	int32_t dataBuff[] = {1,2,3,4,5};
	NetworkMessage::WriteToNetwork(bulkImportMessage, sock);
	NetworkMessage::WriteRaw(sock, reinterpret_cast<char*>(dataBuff), 5, DataType::COLUMN_INT);
	auto response = NetworkMessage::ReadFromNetwork(sock);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ASSERT_TRUE(response.UnpackTo(&infoMessage));
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
}

TEST(TCPServer, ServerMessageInfo)
{
	try
	{
		printf("\nServerMessageInfo\n");
		ITCPWorker::ResetGlobalQuitFlag();
		TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345);
		auto future = std::thread([&testServer]() {testServer.Run(); });
		boost::asio::io_context context;
		auto sock = connectSocketToTestServer(context);
		ASSERT_NO_THROW(connect(sock));
		ASSERT_NO_THROW(disconnect(sock));
		testServer.Abort();
		future.join();
		printf("\nServerMessageInfoEnd\n");
	}
	catch (...)
	{
		ASSERT_TRUE(false);
	}
}

TEST(TCPServer, ServerMessageSetDB)
{
	try
	{
		printf("\nServerMessageSetDB\n");
		ITCPWorker::ResetGlobalQuitFlag();
		TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345);
		auto future = std::thread([&testServer]() {testServer.Run(); });
		boost::asio::io_context context;
		auto sock = connectSocketToTestServer(context);
		ASSERT_NO_THROW(connect(sock));
		ASSERT_NO_THROW(setDatabase(sock,"test"));
		ASSERT_NO_THROW(disconnect(sock));
		testServer.Abort();
		future.join();
		printf("\nServerMessageSetDBEnd\n");
	}
	catch (...)
	{
		ASSERT_TRUE(false);
	}
}

TEST(TCPServer, ServerMessageQuery)
{
	try
	{
		printf("\nServerMessageQuery\n");
		ITCPWorker::ResetGlobalQuitFlag();
		TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345);
		auto future = std::thread([&testServer]() {testServer.Run(); });
		boost::asio::io_context context;
		auto sock = connectSocketToTestServer(context);
		ASSERT_NO_THROW(connect(sock));
		ASSERT_NO_THROW(query(sock, "test"));
		ColmnarDB::NetworkClient::Message::QueryResponseMessage resp;
		ASSERT_NO_THROW(resp = getNextQueryResult(sock));
		ASSERT_EQ(resp.payloads().at("test").stringpayload().stringdata()[0], "test");
		ASSERT_NO_THROW(disconnect(sock));
		testServer.Abort();
		future.join();
		printf("\nServerMessageQueryEnd\n");
	}
	catch (...)
	{
		ASSERT_TRUE(false);
	}
}

TEST(TCPServer, ServerMessageCSV)
{
	try
	{
		printf("\nServerMessageCSV\n");
		ITCPWorker::ResetGlobalQuitFlag();
		TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345);
		auto future = std::thread([&testServer]() {testServer.Run(); });
		boost::asio::io_context context;
		auto sock = connectSocketToTestServer(context);
		ASSERT_NO_THROW(connect(sock));
		ASSERT_NO_THROW(importCSV(sock, "test","test"));
		ASSERT_NO_THROW(disconnect(sock));
		testServer.Abort();
		future.join();
		printf("\nServerMessageCSVEnd\n");
	}
	catch (...)
	{
		ASSERT_TRUE(false);
	}
}


TEST(TCPServer, ServerMessageBulkImport)
{
	try
	{
		printf("\nServerMessageBulkImport\n");
		ITCPWorker::ResetGlobalQuitFlag();
		TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345);
		auto future = std::thread([&testServer]() {testServer.Run(); });
		boost::asio::io_context context;
		auto sock = connectSocketToTestServer(context);
		ASSERT_NO_THROW(connect(sock));
		ASSERT_NO_THROW(bulkImport(sock));
		ASSERT_NO_THROW(disconnect(sock));
		testServer.Abort();
		future.join();
		printf("\nServerMessageBulkImportEnd\n");
	}
	catch (...)
	{
		ASSERT_TRUE(false);
	}
}