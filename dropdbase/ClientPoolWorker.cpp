#include "ClientPoolWorker.h"
#include "NetworkMessage.h"
#include "messages/InfoMessage.pb.h"
#include "messages/QueryMessage.pb.h"
#include "messages/CSVImportMessage.pb.h"
#include "messages/SetDatabaseMessage.pb.h"
#include "IClientHandler.h"
#include <boost/log/trivial.hpp>

ClientPoolWorker::ClientPoolWorker(std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout) 
	: ITCPWorker(std::move(clientHandler), std::move(socket), requestTimeout)
{
	quit_ = false;
}

/// <summary>
/// Main client request handler
/// </summary>
void ClientPoolWorker::HandleClient()
{
	BOOST_LOG_TRIVIAL(debug) << "Waiting for hello from " << socket_.remote_endpoint().address().to_string() << "\n";
	auto recvMsg = NetworkMessage::ReadFromNetwork(socket_);
	ColmnarDB::NetworkClient::Message::InfoMessage outInfo;
	if (!recvMsg.UnpackTo(&outInfo))
	{
		BOOST_LOG_TRIVIAL(error) << "Invalid message received from client " << socket_.remote_endpoint().address().to_string() << "\n";
		Abort();
		return;
	}
	BOOST_LOG_TRIVIAL(debug) << "Hello from " << socket_.remote_endpoint().address().to_string() << "\n";
	outInfo.set_message("");
	outInfo.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
	NetworkMessage::WriteToNetwork(outInfo, socket_);
	BOOST_LOG_TRIVIAL(debug) << "Hello to " << socket_.remote_endpoint().address().to_string() << "\n";
	try
	{
		while (!quit_)
		{
			BOOST_LOG_TRIVIAL(debug) << "Waiting for message from " << socket_.remote_endpoint().address().to_string() << "\n";
			recvMsg = NetworkMessage::ReadFromNetwork(socket_);
			BOOST_LOG_TRIVIAL(debug) << "Got message from " << socket_.remote_endpoint().address().to_string() << "\n";
			ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
			ColmnarDB::NetworkClient::Message::QueryMessage queryMessage;
			ColmnarDB::NetworkClient::Message::CSVImportMessage csvImportMessage;
			ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
			if (recvMsg.UnpackTo(&infoMessage))
			{
				BOOST_LOG_TRIVIAL(debug) << "Info message from " << socket_.remote_endpoint().address().to_string() << "\n";
				std::unique_ptr<google::protobuf::Message> resultMessage = clientHandler_->HandleInfoMessage(*this, infoMessage);
				
				if (resultMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*resultMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&queryMessage))
			{
				BOOST_LOG_TRIVIAL(debug) << "Query message from " << socket_.remote_endpoint().address().to_string() << "\n";
				std::unique_ptr<google::protobuf::Message> waitMessage = clientHandler_->HandleQuery(*this, queryMessage);
				if (waitMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*waitMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&csvImportMessage))
			{
				BOOST_LOG_TRIVIAL(debug) << "CSV message from " << socket_.remote_endpoint().address().to_string() << "\n";
				std::unique_ptr<google::protobuf::Message> importResultMessage = clientHandler_->HandleCSVImport(*this, csvImportMessage);
				if (importResultMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*importResultMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&setDatabaseMessage))
			{
				BOOST_LOG_TRIVIAL(debug) << "Set database message from " << socket_.remote_endpoint().address().to_string() << "\n";
				std::unique_ptr<google::protobuf::Message> setDatabaseResult = clientHandler_->HandleSetDatabase(*this, setDatabaseMessage);
				if (setDatabaseResult != nullptr)
				{
					NetworkMessage::WriteToNetwork(*setDatabaseResult, socket_);
				}
			}
			else
			{
				BOOST_LOG_TRIVIAL(error) << "Invalid message from " << socket_.remote_endpoint().address().to_string() << "\n";
			}
		}
	}
	catch (std::exception& e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what() << std::endl;
	}
	if (!quit_)
	{
		Abort();
	}
}

/// <summary>
/// Stops the worker after reading next message
/// </summary>
void ClientPoolWorker::Abort()
{
	quit_ = true;
	socket_.close();
}
