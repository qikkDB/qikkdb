#include "ClientPoolWorker.h"
#include "NetworkMessage.h"
#include "messages/InfoMessage.pb.h"
#include "messages/QueryMessage.pb.h"
#include "messages/CSVImportMessage.pb.h"
#include "messages/SetDatabaseMessage.pb.h"
#include "IClientHandler.h"


ClientPoolWorker::ClientPoolWorker(std::set<std::shared_ptr<ITCPWorker>>& activeWorkers, std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout) 
	: ITCPWorker(activeWorkers, std::move(clientHandler), std::move(socket), requestTimeout)
{
	quit_ = false;
}

/// <summary>
/// Main client request handler
/// </summary>
void ClientPoolWorker::HandleClient()
{
	auto recvMsg = NetworkMessage::ReadFromNetwork(socket_);
	ColmnarDB::NetworkClient::Message::InfoMessage outInfo;
	if (!recvMsg.UnpackTo(&outInfo))
	{
		Abort();
		return;
	}
	outInfo.set_message("");
	outInfo.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
	NetworkMessage::WriteToNetwork(outInfo, socket_);
	try
	{
		while (!quit_)
		{
			recvMsg = NetworkMessage::ReadFromNetwork(socket_);
			ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
			ColmnarDB::NetworkClient::Message::QueryMessage queryMessage;
			ColmnarDB::NetworkClient::Message::CSVImportMessage csvImportMessage;
			ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
			if (recvMsg.UnpackTo(&infoMessage))
			{
				std::unique_ptr<google::protobuf::Message> resultMessage = clientHandler_->HandleInfoMessage(*this, infoMessage);
				
				if (resultMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*resultMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&queryMessage))
			{
				std::unique_ptr<google::protobuf::Message> waitMessage = clientHandler_->HandleQuery(*this, queryMessage);
				if (waitMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*waitMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&csvImportMessage))
			{
				std::unique_ptr<google::protobuf::Message> importResultMessage = clientHandler_->HandleCSVImport(*this, csvImportMessage);
				if (importResultMessage != nullptr)
				{
					NetworkMessage::WriteToNetwork(*importResultMessage, socket_);
				}
			}
			else if (recvMsg.UnpackTo(&csvImportMessage))
			{
				std::unique_ptr<google::protobuf::Message> setDatabaseResult = clientHandler_->HandleSetDatabase(*this, setDatabaseMessage);
				if (setDatabaseResult != nullptr)
				{
					NetworkMessage::WriteToNetwork(*setDatabaseResult, socket_);
				}
			}
			else
			{
				std::cerr << "Invalid message received";
			}
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
}

/// <summary>
/// Stops the worker after reading next message
/// </summary>
void ClientPoolWorker::Abort()
{
	quit_ = true;
	socket_.close();
	ITCPWorker::Abort();
}
